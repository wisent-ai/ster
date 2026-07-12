"""GROM steering object creation helper."""
from __future__ import annotations
import torch

from wisent.core.utils.config_tools.constants import (
    RECURSION_INITIAL_DEPTH,
    COMBO_OFFSET,
    AXIS_ROWS,
    SS_GROM_BEHAVIOR_TARGET,
)


def _require_arg(args, attr_name):
    val = getattr(args, attr_name, None)
    if val is None:
        raise ValueError(
            f"Parameter '{attr_name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return val


def _grom_training_step(
    gate_network, sensor_pos, sensor_neg, gate_temperature,
    gate_threshold, layer_order, directions, direction_weights,
    all_pos, all_neg, retain_weight,
):
    """Single GROM training step — computes total loss."""
    total_loss = torch.zeros(())
    pos_gate = gate_network(sensor_pos, gate_temperature)
    neg_gate = gate_network(sensor_neg, gate_temperature)
    gate_loss = (
        torch.relu(gate_threshold - pos_gate).mean()
        + torch.relu(neg_gate - gate_threshold).mean()
    )
    total_loss = total_loss + gate_loss
    for layer in layer_order:
        if layer not in directions:
            continue
        dirs = torch.nn.functional.normalize(directions[layer], dim=COMBO_OFFSET)
        weights = torch.softmax(direction_weights[layer], dim=AXIS_ROWS)
        effective_dir = (weights.unsqueeze(-COMBO_OFFSET) * dirs).sum(dim=AXIS_ROWS)
        pos_proj = (all_pos[layer] * effective_dir).sum(dim=COMBO_OFFSET)
        neg_proj = (all_neg[layer] * effective_dir).sum(dim=COMBO_OFFSET).abs()
        total_loss = total_loss + torch.relu(SS_GROM_BEHAVIOR_TARGET - pos_proj).mean()
        total_loss = total_loss + neg_proj.mean() * retain_weight
    return total_loss


def _create_grom_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
    log_interval: int,
    *,
    gate_dim_min: int,
    gate_dim_max: int,
    gate_dim_divisor: int,
    gate_shrink_factor: int,
    intensity_dim_min: int,
    intensity_dim_max: int,
    intensity_dim_divisor: int,
    create_noise_scale: float,
    create_gate_threshold: float,
) -> GROMSteeringObject:
    """Create GROM steering object with learned networks."""
    from wisent.core.control.steering_methods.methods.grom import GROMMethod
    from wisent.core.control.steering_methods._steering_object_grom import GROMGateNetwork, GROMIntensityNetwork
    
    num_directions = _require_arg(args, 'grom_num_directions')
    hidden_dim = metadata.hidden_dim
    num_layers = len(available_layers)
    
    # Determine sensor layer — use last available layer if not specified
    sensor_layer_idx = getattr(args, 'grom_sensor_layer', None)
    if sensor_layer_idx is None:
        sensor_layer_idx = num_layers - 1
    sensor_layer = int(available_layers[min(sensor_layer_idx, num_layers - 1)])
    
    gate_hidden_dim = max(gate_dim_min, min(gate_dim_max, hidden_dim // gate_dim_divisor))
    intensity_hidden_dim = max(intensity_dim_min, min(intensity_dim_max, hidden_dim // intensity_dim_divisor))
    max_alpha = _require_arg(args, 'grom_max_alpha')
    
    # Initialize networks
    gate_network = GROMGateNetwork(hidden_dim, gate_hidden_dim, shrink_factor=gate_shrink_factor)
    intensity_network = GROMIntensityNetwork(hidden_dim, num_layers, intensity_hidden_dim, max_alpha)
    
    # Prepare data
    layer_order = [int(l) for l in available_layers]
    
    # Initialize directions and weights
    directions = {}
    direction_weights = {}
    
    # Collect all data
    all_pos = {}
    all_neg = {}
    
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        
        if not pos_list or not neg_list:
            continue
        
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        layer_int = int(layer_str)
        all_pos[layer_int] = pos_tensor
        all_neg[layer_int] = neg_tensor
        
        # Initialize directions with CAA + perturbations
        caa_dir = torch.nn.functional.normalize(pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0), dim=0)
        dirs = torch.randn(num_directions, hidden_dim)
        dirs[0] = caa_dir
        for i in range(1, num_directions):
            dirs[i] = torch.nn.functional.normalize(caa_dir + torch.randn(hidden_dim) * create_noise_scale, dim=0)
        
        directions[layer_int] = torch.nn.Parameter(dirs)
        direction_weights[layer_int] = torch.nn.Parameter(torch.zeros(num_directions))
    
    # Joint optimization
    all_params = list(directions.values()) + list(direction_weights.values())
    all_params.extend(gate_network.parameters())
    all_params.extend(intensity_network.parameters())
    
    learning_rate = _require_arg(args, 'grom_learning_rate')
    weight_decay = _require_arg(args, 'grom_weight_decay')
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
    
    sensor_pos = all_pos[sensor_layer]
    sensor_neg = all_neg[sensor_layer]
    
    retain_weight = _require_arg(args, 'grom_retain_weight')
    max_grad_norm = _require_arg(args, 'grom_max_grad_norm')

    gate_temperature = _require_arg(args, 'grom_gate_temperature')

    print(f"   Training GROM ({num_directions} directions, {len(layer_order)} layers)...")

    best_loss = float('inf')
    best_state = None
    optimization_steps = _require_arg(args, 'grom_optimization_steps')
    for step in range(optimization_steps):
        optimizer.zero_grad()

        total_loss = _grom_training_step(
            gate_network, sensor_pos, sensor_neg, gate_temperature,
            create_gate_threshold, layer_order, directions,
            direction_weights, all_pos, all_neg, retain_weight,
        )
        current_loss = total_loss.item()
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"      Step {step}: NaN/Inf detected, reverting to best state")
            break

        total_loss.backward()
        for p in all_params:
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad)
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)
        optimizer.step()

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                "directions": {l: p.detach().clone() for l, p in directions.items()},
                "dw": {l: p.detach().clone() for l, p in direction_weights.items()},
                "gate": {key: value.detach().clone() for key, value in gate_network.state_dict().items()},
                "intensity": {key: value.detach().clone() for key, value in intensity_network.state_dict().items()},
            }
        if step % log_interval == RECURSION_INITIAL_DEPTH:
            print(f"      Step {step}: loss={current_loss:.4f}")

    if best_state is not None:
        for l in directions:
            directions[l].data.copy_(best_state["directions"][l])
            direction_weights[l].data.copy_(best_state["dw"][l])
        gate_network.load_state_dict(best_state["gate"])
        intensity_network.load_state_dict(best_state["intensity"])
    
    # Finalize directions
    final_directions = {}
    final_weights = {}
    
    for layer in layer_order:
        if layer in directions:
            final_directions[layer] = torch.nn.functional.normalize(directions[layer].detach(), dim=1)
            final_weights[layer] = torch.softmax(direction_weights[layer].detach(), dim=0)
    
    with torch.no_grad():
        pos_gate = gate_network(sensor_pos, gate_temperature)
        neg_gate = gate_network(sensor_neg, gate_temperature)

    print(f"   Sensor layer: {sensor_layer}")
    print(f"   Final gate accuracy: pos={pos_gate.mean().item():.3f}, neg={neg_gate.mean().item():.3f}")
    
    from wisent.core.control.steering_methods._steering_object_grom import GROMSteeringObject
    return GROMSteeringObject(
        metadata=metadata,
        directions=final_directions,
        direction_weights=final_weights,
        gate_network=gate_network,
        intensity_network=intensity_network,
        layer_order=layer_order,
        gate_temperature=gate_temperature,
        max_alpha=max_alpha,
    )
