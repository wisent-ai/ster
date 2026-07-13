"""GROM runtime hooks and weight modification."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Callable

from wisent.core.utils.config_tools.constants import SEPARATOR_WIDTH_STANDARD
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from torch.nn import Module

_LOG = setup_logger(__name__)


class GROMRuntimeHooks:
    """Runtime hook system for GROM sensor-aware dynamic steering."""

    def __init__(
        self,
        model: Module,
        grom_result,
        base_strength: float,
        gate_threshold: float | None = None,
        use_soft_gating: bool = True,
        strength_provider: Callable[[torch.Tensor], float] | None = None,
    ) -> None:
        self.model = model
        self.grom_result = grom_result
        self.base_strength = base_strength
        self.gate_threshold = gate_threshold
        self.use_soft_gating = use_soft_gating
        self.strength_provider = strength_provider
        self._hooks = []
        self._sensor_activation = None
        self._current_gate = None
        self._current_intensities = None
        self._current_strategy_strength = 1.0
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._layers = model.transformer.h
        elif hasattr(model, "layers"):
            self._layers = model.layers
        else:
            raise ValueError("GROM model has no supported decoder layers")

        self._layer_name_to_idx = {
            layer_name: self._model_index(layer_name, "steering layer")
            for layer_name in grom_result.layer_order
        }
        sensor_value = getattr(grom_result, "sensor_layer", None)
        metadata = getattr(grom_result, "metadata", None)
        metadata_sensor = (
            metadata.get("sensor_layer")
            if isinstance(metadata, dict)
            else getattr(metadata, "sensor_layer", None)
        )
        metadata_component = (
            metadata.get("extraction_component")
            if isinstance(metadata, dict)
            else getattr(metadata, "extraction_component", None)
        )
        if metadata_component != "residual_stream":
            raise ValueError(
                "GROM requires extraction_component='residual_stream'"
            )
        if sensor_value is None:
            sensor_value = metadata_sensor
        elif metadata_sensor is not None and self._layer_number(sensor_value) != self._layer_number(metadata_sensor):
            raise ValueError("GROM sensor_layer conflicts with metadata.sensor_layer")
        if sensor_value is None:
            raise ValueError("GROM sensor_layer is required")
        self._sensor_layer_idx = self._model_index(sensor_value, "sensor_layer")
        if (
            self._layer_name_to_idx
            and self._sensor_layer_idx >= min(self._layer_name_to_idx.values())
        ):
            raise ValueError(
                "GROM sensor_layer must be strictly earlier than every steering layer"
            )
        if not use_soft_gating and gate_threshold is None:
            raise ValueError("GROM gate_threshold is required for hard gating")

    @staticmethod
    def _layer_number(layer) -> int:
        try:
            number = int(str(layer).split("_")[-1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid GROM layer identifier: {layer!r}") from exc
        if number <= 0:
            raise ValueError(f"GROM sensor_layer/steering layer must be >= 1, got {number}")
        return number

    def _model_index(self, layer, label: str) -> int:
        number = self._layer_number(layer)
        if number > len(self._layers):
            raise ValueError(
                f"GROM {label} {number} exceeds model layer count {len(self._layers)}"
            )
        return number - 1

    def install(self) -> None:
        """Install sensor first, then steering hooks in causal layer order."""
        self.remove()
        self._hooks.append(
            self._layers[self._sensor_layer_idx].register_forward_hook(self._sensor_hook)
        )
        for layer_name in self.grom_result.layer_order:
            layer_idx = self._layer_name_to_idx[layer_name]
            steering_hook = self._layers[layer_idx].register_forward_hook(
                lambda module, input, output, ln=layer_name: self._steering_hook(
                    module, input, output, ln
                )
            )
            self._hooks.append(steering_hook)

    def begin_generation(self) -> None:
        """Reset per-request gate and absolute generated-token schedule state."""
        self._sensor_activation = None
        self._current_gate = None
        self._current_intensities = None
        self._current_strategy_strength = 1.0
        reset = getattr(self.strength_provider, "reset", None)
        if callable(reset):
            reset()

    def remove(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.begin_generation()

    def _sensor_hook(self, module, input, output):
        """Compute gate and all intensities from the exact sensor activation."""
        hidden_states = output[0] if isinstance(output, tuple) else output
        sensor_h = hidden_states[:, -1, :] if hidden_states.dim() == 3 else hidden_states
        self._sensor_activation = sensor_h.detach()
        self._current_strategy_strength = (
            self.strength_provider(hidden_states)
            if self.strength_provider is not None
            else 1.0
        )
        with torch.no_grad():
            gate_value = self.grom_result.predict_gate(sensor_h)
            if self.use_soft_gating:
                self._current_gate = gate_value
            else:
                self._current_gate = (gate_value > self.gate_threshold).float()
            self._current_intensities = self.grom_result.predict_intensity(sensor_h)
        return output

    def _steering_hook(self, module, input, output, layer_name):
        """Apply dynamic steering using the sensor-derived gate and intensities."""
        if self._current_gate is None or self._current_intensities is None:
            return output
        hidden_states = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        direction = self.grom_result.get_effective_direction(layer_name).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        if layer_name not in self._current_intensities:
            raise KeyError(f"No predicted intensity for GROM layer {layer_name!r}")
        intensity = self._current_intensities[layer_name].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        gate = self._current_gate.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        strategy_strength = self._current_strategy_strength
        if hidden_states.dim() == 3:
            gate = gate.reshape(-1, 1, 1)
            intensity = intensity.reshape(-1, 1, 1)
            direction = direction.view(1, 1, -1)
        elif hidden_states.dim() == 2:
            gate = gate.reshape(-1, 1)
            intensity = intensity.reshape(-1, 1)
            direction = direction.view(1, -1)
        steering_delta = (
            gate
            * intensity
            * self.base_strength
            * strategy_strength
            * direction
        )
        hidden_states = hidden_states + steering_delta
        return (hidden_states,) + rest if rest is not None else hidden_states

    def get_current_gate(self) -> float | None:
        """Get the current gate value."""
        return self._current_gate.mean().item() if self._current_gate is not None else None

    def get_current_intensities(self) -> dict | None:
        """Get current per-layer intensities."""
        if self._current_intensities is None:
            return None
        return {k: v.mean().item() for k, v in self._current_intensities.items()}


def project_weights_grom(
    model: Module, grom_result, base_strength: float, base_layer_weight: float,
    components: list[str] | None = None,
    use_learned_intensities: bool = True, verbose: bool = True,
) -> dict[str, int]:
    """Bake GROM effective directions into model weights using ADDITIVE steering."""
    from wisent.core.weight_modification.methods.additive import bake_steering_into_weights
    from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
    if base_layer_weight is None:
        raise MissingParameterError(params=["base_layer_weight"], context="project_weights_grom requires base_layer_weight")
    log = bind(_LOG, num_layers=len(grom_result.directions))
    if components is None:
        components = ["self_attn.o_proj", "mlp.down_proj"]
    effective_vectors, layer_weights = {}, {}
    for layer_name in grom_result.layer_order:
        eff_dir = grom_result.get_effective_direction(layer_name)
        try:
            layer_idx = int(str(layer_name).split("_")[-1])
        except (ValueError, IndexError):
            continue
        effective_vectors[layer_idx] = eff_dir
        if use_learned_intensities:
            dir_weights = grom_result.direction_weights.get(layer_name)
            if dir_weights is None:
                raise KeyError(f"No direction_weights for '{layer_name}'")
            weight = base_layer_weight + (dir_weights.max() - dir_weights.min()).item()
            layer_weights[layer_idx] = weight
    if verbose:
        print(f"\n{'='*SEPARATOR_WIDTH_STANDARD}\nGROM WEIGHT MODIFICATION (ADDITIVE)\n{'='*SEPARATOR_WIDTH_STANDARD}")
        print(f"Layers: {len(effective_vectors)}, Components: {components}, Base strength: {base_strength}\n{'='*SEPARATOR_WIDTH_STANDARD}\n")
    weighted_vectors = {layer_idx: vec * layer_weights[layer_idx] if use_learned_intensities else vec
                        for layer_idx, vec in effective_vectors.items()}
    steering_vectors = LayerActivations(weighted_vectors)
    stats = bake_steering_into_weights(model=model, steering_vectors=steering_vectors, alpha=base_strength, method="bias", components=components, verbose=verbose)
    stats["grom_layers"] = len(grom_result.layer_order)
    stats["grom_directions_per_layer"] = grom_result.directions[grom_result.layer_order[0]].shape[0]
    stats["method"] = "additive"
    return stats
