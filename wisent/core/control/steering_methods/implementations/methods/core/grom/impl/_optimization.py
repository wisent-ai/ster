"""GROM train_grom method and joint optimization loop."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.utils.config_tools.constants import RECURSION_INITIAL_DEPTH, COMBO_OFFSET, SCORE_RANGE_MIN
from wisent.core.control.steering_methods.methods.grom._config import (
    GatingNetwork,
    IntensityNetwork,
    GeometryAdaptation,
)


def _require_finite_tensor(value: torch.Tensor, name: str, *, step: Optional[int] = None) -> None:
    """Reject non-finite optimization state with GROM-specific context."""
    finite_mask = torch.isfinite(value.detach())
    if bool(finite_mask.all()):
        return
    non_finite_count = value.numel() - int(finite_mask.sum().item())
    step_context = "" if step is None else f" at optimization step {step}"
    raise RuntimeError(
        f"GROM numerical invariant failed{step_context}: {name} contains "
        f"{non_finite_count}/{value.numel()} non-finite values"
    )


def _require_finite_config(name: str, value: float, *, positive: bool = False) -> None:
    """Validate scalar optimizer configuration before constructing optimizer state."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"GROM {name} must be a finite number; got {value!r}") from exc
    if not math.isfinite(numeric_value) or (positive and numeric_value <= 0.0):
        requirement = "finite and greater than zero" if positive else "finite"
        raise ValueError(f"GROM {name} must be {requirement}; got {value!r}")


def train_grom_impl(self, pair_set: ContrastivePairSet):
    """Train all GROM components on the exact configured sensor and steering layers."""
    from wisent.core.control.steering_methods.methods.grom.grom import GROMResult

    buckets = self._collect_from_set(pair_set)
    if not buckets:
        raise InsufficientDataError(reason="No valid activation pairs found")

    available_by_index = {}
    for layer_name in buckets:
        try:
            layer_index = int(str(layer_name).split("_")[-1])
        except (ValueError, IndexError) as exc:
            raise InsufficientDataError(
                reason=f"Unparseable activation layer name: {layer_name!r}"
            ) from exc
        if layer_index in available_by_index:
            raise InsufficientDataError(
                reason=f"Multiple activation names resolve to layer {layer_index}"
            )
        available_by_index[layer_index] = layer_name

    if self.config.steering_layers is None or self.config.sensor_layer is None:
        detected_num_layers = max(available_by_index) + COMBO_OFFSET
        self.config.resolve_layers(detected_num_layers)

    layer_names = []
    hidden_dim = None
    for layer_index in self.config.steering_layers:
        if layer_index not in available_by_index:
            raise InsufficientDataError(
                reason=f"Missing configured steering layer {layer_index}"
            )
        layer_name = available_by_index[layer_index]
        pos_list, neg_list = buckets[layer_name]
        if not pos_list or not neg_list:
            raise InsufficientDataError(
                reason=f"Empty activations at steering layer {layer_index}"
            )
        candidate_dim = pos_list[0].reshape(-1).shape[0]
        if hidden_dim is None:
            hidden_dim = candidate_dim
        elif candidate_dim != hidden_dim:
            raise InsufficientDataError(reason="Steering layers have different hidden dimensions")
        layer_names.append(layer_name)
    if not layer_names or hidden_dim is None:
        raise InsufficientDataError(reason="No valid steering layers found")

    if self.config.sensor_layer not in available_by_index:
        raise InsufficientDataError(
            reason=f"Missing configured sensor layer {self.config.sensor_layer}"
        )
    sensor_layer = available_by_index[self.config.sensor_layer]
    sensor_pos, sensor_neg = buckets[sensor_layer]
    if not sensor_pos or not sensor_neg:
        raise InsufficientDataError(
            reason=f"Empty activations at sensor layer {self.config.sensor_layer}"
        )
    if sensor_pos[0].reshape(-1).shape[0] != hidden_dim:
        raise InsufficientDataError(reason="Sensor and steering hidden dimensions differ")

    self.config.resolve_network_dims(hidden_dim)
    num_layers = len(layer_names)

    geometry_adaptation = None
    effective_num_directions = self.config.num_directions
    enable_gating = True
    if self.config.adapt_to_geometry:
        geometry_adaptation = self._analyze_and_adapt_geometry(
            buckets, layer_names, hidden_dim, default_score=SCORE_RANGE_MIN,
        )
        effective_num_directions = geometry_adaptation.adapted_num_directions
        enable_gating = geometry_adaptation.gating_enabled

    directions = self._initialize_directions(
        buckets, layer_names, hidden_dim,
        num_directions=effective_num_directions,
    )
    gate_network: Optional[GatingNetwork] = None
    if enable_gating:
        gate_network = GatingNetwork(
            hidden_dim, self.config.gate_hidden_dim,
            shrink_factor=self.config.gate_shrink_factor,
        )
    intensity_network = IntensityNetwork(
        hidden_dim, num_layers, self.config.intensity_hidden_dim,
        self.config.max_alpha,
    )
    direction_weight_params = {
        layer: nn.Parameter(torch.zeros(effective_num_directions))
        for layer in layer_names
    }
    data_layers = list(layer_names)
    if sensor_layer not in data_layers:
        data_layers.append(sensor_layer)
    data = self._prepare_data_tensors(buckets, data_layers)
    directions, gate_network, intensity_network, direction_weights = self._joint_optimization(
        directions=directions,
        gate_network=gate_network,
        intensity_network=intensity_network,
        direction_weight_params=direction_weight_params,
        data=data,
        layer_names=layer_names,
        sensor_layer=sensor_layer,
        log_interval=self.log_interval,
        enable_gating=enable_gating,
    )
    return GROMResult(
        directions=directions,
        gate_network=gate_network,
        intensity_network=intensity_network,
        direction_weights=direction_weights,
        layer_order=layer_names,
        sensor_layer=sensor_layer,
        gate_temperature=self.config.gate_temperature,
        metadata={
            "config": self.config.__dict__,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "sensor_layer": sensor_layer,
            "training_logs": self._training_logs,
            "effective_num_directions": effective_num_directions,
            "gating_enabled": enable_gating,
        },
        geometry_adaptation=geometry_adaptation,
    )


def _joint_optimization_impl(
    self,
    directions: Dict[LayerName, torch.Tensor],
    gate_network: Optional[GatingNetwork],
    intensity_network: IntensityNetwork,
    direction_weight_params: Dict[LayerName, nn.Parameter],
    data: Dict[str, Dict[LayerName, torch.Tensor]],
    layer_names: List[LayerName],
    sensor_layer: LayerName,
    log_interval: int,
    enable_gating: bool = True,
) -> Tuple[Dict[LayerName, torch.Tensor], Optional[GatingNetwork], IntensityNetwork, Dict[LayerName, torch.Tensor]]:
    """
    Joint end-to-end optimization of all GROM components.
    """
    _require_finite_config("learning_rate", self.config.learning_rate)
    _require_finite_config("weight_decay", self.config.weight_decay)
    _require_finite_config("eta_min_factor", self.config.eta_min_factor)
    _require_finite_config("max_grad_norm", self.config.max_grad_norm, positive=True)
    if gate_network is not None:
        _require_finite_config("gate_temperature", self.config.gate_temperature, positive=True)
    for polarity, layer_data in data.items():
        for layer, activations in layer_data.items():
            _require_finite_tensor(
                activations,
                f"{polarity} activation data for layer {layer!r}",
            )
    # Make directions trainable
    direction_params = {layer: nn.Parameter(dirs.clone()) for layer, dirs in directions.items()}
    # Collect all parameters
    all_params = []
    all_params.extend(direction_params.values())
    if gate_network is not None:
        all_params.extend(gate_network.parameters())
    all_params.extend(intensity_network.parameters())
    all_params.extend(direction_weight_params.values())
    for parameter_index, parameter in enumerate(all_params):
        _require_finite_tensor(parameter, f"initial parameter {parameter_index}")
    optimizer = torch.optim.AdamW(all_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=self.config.optimization_steps, eta_min=self.config.learning_rate * self.config.eta_min_factor
    )
    best_loss = float('inf')
    best_state = None
    for step in range(self.config.optimization_steps):
        optimizer.zero_grad()
        # Compute effective directions (weighted sum)
        effective_dirs = {}
        for layer in layer_names:
            weights = F.softmax(direction_weight_params[layer], dim=0)
            dirs = direction_params[layer]
            dirs_norm = F.normalize(dirs, p=2, dim=1)
            effective_dirs[layer] = (weights.unsqueeze(-1) * dirs_norm).sum(dim=0)
        # Get sensor layer data
        pos_sensor = data["pos"][sensor_layer]
        neg_sensor = data["neg"][sensor_layer]
        # Predict gates (or use constant 1.0 if gating disabled)
        if gate_network is not None:
            pos_gate = gate_network(pos_sensor, self.config.gate_temperature)
            neg_gate = gate_network(neg_sensor, self.config.gate_temperature)
        else:
            pos_gate = torch.ones(pos_sensor.shape[0], device=pos_sensor.device)
            neg_gate = torch.ones(neg_sensor.shape[0], device=neg_sensor.device)
        # Predict intensities
        pos_intensity = intensity_network(pos_sensor)  # [N_pos, num_layers]
        neg_intensity = intensity_network(neg_sensor)  # [N_neg, num_layers]
        # Compute losses
        loss, loss_components = self._compute_grom_loss(
            direction_params=direction_params,
            effective_dirs=effective_dirs,
            pos_gate=pos_gate, neg_gate=neg_gate,
            pos_intensity=pos_intensity, neg_intensity=neg_intensity,
            data=data, layer_names=layer_names, step=step,
            direction_weight_params=direction_weight_params,
            contrastive_margin=self.config.contrastive_margin,
            contrastive_weight=self.config.contrastive_weight,
            utility_weight=self.config.utility_weight,
            concentration_weight=self.config.concentration_weight,
            gate_warmup_weight=self.config.gate_warmup_weight,
            caa_alignment_weight=self.config.caa_alignment_weight,
        )
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
            component_summary = ", ".join(
                f"{name}={component.detach().item():.6g}"
                for name, component in loss_components.items()
            )
            raise RuntimeError(
                f"GROM optimization step {step} produced a non-finite total loss "
                "before backward/optimizer.step; optimizer state was not advanced. "
                f"Loss components: {component_summary}"
            )
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                all_params,
                max_norm=self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"GROM optimization step {step} produced non-finite gradients; "
                "optimizer.step was not called and optimizer state was not advanced"
            ) from exc
        optimizer.step()
        for parameter_index, parameter in enumerate(all_params):
            _require_finite_tensor(
                parameter,
                f"parameter {parameter_index} after optimizer.step; best_state was not updated",
                step=step,
            )
        for state_index, optimizer_state in enumerate(optimizer.state.values()):
            for state_name, state_value in optimizer_state.items():
                if isinstance(state_value, torch.Tensor):
                    _require_finite_tensor(
                        state_value,
                        f"optimizer state {state_index}.{state_name}; best_state was not updated",
                        step=step,
                    )
        scheduler.step()
        # Apply constraints to directions
        with torch.no_grad():
            for layer in layer_names:
                direction_params[layer].data = self._apply_direction_constraints(
                    direction_params[layer].data
                )
        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                "directions": {l: p.detach().clone() for l, p in direction_params.items()},
                "gate_network": {k: v.detach().clone() for k, v in gate_network.state_dict().items()} if gate_network is not None else None,
                "intensity_network": {k: v.detach().clone() for k, v in intensity_network.state_dict().items()},
                "direction_weights": {l: F.softmax(p.detach().clone(), dim=0) for l, p in direction_weight_params.items()},
            }
        # Log
        if step % log_interval == RECURSION_INITIAL_DEPTH or step == self.config.optimization_steps - COMBO_OFFSET:
            # Compute direction weight statistics
            weight_stds = []
            weight_maxes = []
            for layer in layer_names:
                weights = F.softmax(direction_weight_params[layer], dim=0)
                weight_stds.append(weights.std().item())
                weight_maxes.append(weights.max().item())
            self._training_logs.append({
                "step": step,
                "total_loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                **{k: v.item() for k, v in loss_components.items()},
                "pos_gate_mean": pos_gate.mean().item(),
                "neg_gate_mean": neg_gate.mean().item(),
                "pos_intensity_mean": pos_intensity.mean().item(),
                "neg_intensity_mean": neg_intensity.mean().item(),
                "direction_weight_std_mean": sum(weight_stds) / len(weight_stds),
                "direction_weight_max_mean": sum(weight_maxes) / len(weight_maxes),
            })
    # Restore best state
    if best_state is not None:
        final_directions = best_state["directions"]
        if gate_network is not None and best_state["gate_network"] is not None:
            gate_network.load_state_dict(best_state["gate_network"])
        intensity_network.load_state_dict(best_state["intensity_network"])
        final_weights = best_state["direction_weights"]
    else:
        final_directions = {l: p.detach() for l, p in direction_params.items()}
        final_weights = {l: F.softmax(p.detach(), dim=0) for l, p in direction_weight_params.items()}
    # Final normalization
    if self.config.normalize:
        final_directions = {l: F.normalize(d, p=2, dim=1) for l, d in final_directions.items()}
    # POLARITY CORRECTION: Ensure directions point from neg to pos
    # After optimization, directions may have flipped. We check if pos samples
    # have higher projection than neg samples. If not, flip the direction.
    for layer in layer_names:
        pos_data = data["pos"][layer]
        neg_data = data["neg"][layer]
        # Compute effective direction for this layer
        weights = final_weights[layer]
        dirs = final_directions[layer]
        eff_dir = (weights.unsqueeze(-1) * dirs).sum(dim=0)
        eff_dir = F.normalize(eff_dir, p=2, dim=0)
        # Compute mean projections
        pos_proj = (pos_data * eff_dir).sum(dim=1).mean()
        neg_proj = (neg_data * eff_dir).sum(dim=1).mean()
        # If neg > pos, flip all directions for this layer
        if neg_proj > pos_proj:
            final_directions[layer] = -final_directions[layer]
    return final_directions, gate_network, intensity_network, final_weights
