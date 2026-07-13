"""GROM loss computation, direction constraints, and data collection."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.config_tools.constants import NORM_EPS, STEERING_SCALE_IDENTITY, SS_GROM_BEHAVIOR_TARGET
from wisent.core.utils.infra_tools.errors import InsufficientDataError


def _compute_grom_loss_impl(
    self,
    direction_params: Dict[LayerName, nn.Parameter],
    effective_dirs: Dict[LayerName, torch.Tensor],
    pos_gate: torch.Tensor,
    neg_gate: torch.Tensor,
    pos_intensity: torch.Tensor,
    neg_intensity: torch.Tensor,
    data: Dict[str, Dict[LayerName, torch.Tensor]],
    layer_names: List[LayerName],
    step: int,
    direction_weight_params: Optional[Dict[LayerName, nn.Parameter]] = None,
    *,
    contrastive_margin: float,
    contrastive_weight: float,
    utility_weight: float,
    concentration_weight: float,
    gate_warmup_weight: float,
    caa_alignment_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the full GROM loss.

    Components:
    1. Contrastive loss: CRITICAL - maximize separation between pos and neg projections
    2. Behavior loss: Steering should be effective on positives
    3. Retain loss: Negatives should project LOWER than positives (not orthogonal!)
    4. Sparse loss: Encourage sparse layer activation
    5. Smooth loss: Penalize intensity variance across layers
    6. Independence loss: Directions should be independent
    7. Gate loss: Gate should discriminate pos from neg
    8. Direction utility loss: Differentiate directions based on effectiveness
    9. CAA alignment loss: Keep directions aligned with mean difference
    """
    loss_components = {}

    # 1. CONTRASTIVE SEPARATION LOSS - THE MOST IMPORTANT LOSS
    # This directly maximizes the gap: pos_proj - neg_proj
    contrastive_loss = torch.tensor(0.0)
    for layer in layer_names:
        pos_data = data["pos"][layer]
        neg_data = data["neg"][layer]
        eff_dir = effective_dirs[layer]

        # Compute projections (dot product with direction)
        pos_proj = (pos_data * eff_dir).sum(dim=1)  # [N_pos]
        neg_proj = (neg_data * eff_dir).sum(dim=1)  # [N_neg]

        # We want: pos_proj >> neg_proj (with margin)
        # Use pairwise margin loss: max(0, margin - (pos_proj - neg_proj))
        # Since we may have different numbers of pos/neg, use mean comparison
        pos_mean = pos_proj.mean()
        neg_mean = neg_proj.mean()

        # Margin should be proportional to the scale of activations
        # Typical projections are in range [-5, 5], so margin of 2.0 is reasonable
        margin = contrastive_margin
        contrastive_loss = contrastive_loss + F.relu(margin - (pos_mean - neg_mean))

    contrastive_loss = contrastive_loss / len(layer_names)
    loss_components["contrastive"] = contrastive_loss

    # 2. Behavior loss - optimize the same gate * intensity * direction update
    # applied at inference, so gradients reach every learned component.
    behavior_loss = pos_intensity.new_zeros(())
    for layer_index, layer in enumerate(layer_names):
        pos_data = data["pos"][layer]
        eff_dir = F.normalize(effective_dirs[layer], p=2, dim=-1)
        steering_delta = (
            pos_gate.unsqueeze(-1)
            * pos_intensity[:, layer_index].unsqueeze(-1)
            * eff_dir.unsqueeze(0)
        )
        steered_pos = pos_data + steering_delta
        pos_proj = (steered_pos * eff_dir).sum(dim=1)
        behavior_loss = behavior_loss + F.relu(
            SS_GROM_BEHAVIOR_TARGET - pos_proj
        ).mean()
    behavior_loss = behavior_loss / len(layer_names)
    loss_components["behavior"] = behavior_loss

    # 3. Retain loss - negatives should have LOWER projection than positives
    # Changed from abs() to direct negative projection encouragement
    retain_loss = torch.tensor(0.0)
    for layer in layer_names:
        neg_data = data["neg"][layer]
        eff_dir = effective_dirs[layer]
        neg_proj = (neg_data * eff_dir).sum(dim=1)
        # We want neg_proj < 0 (negative side of direction)
        retain_loss = retain_loss + F.relu(neg_proj).mean()

    retain_loss = retain_loss / len(layer_names)
    loss_components["retain"] = retain_loss

    # 4. Sparse loss - minimizing entropy encourages concentrated layer use.
    pos_intensity_norm = pos_intensity / (
        pos_intensity.sum(dim=1, keepdim=True) + NORM_EPS
    )
    sparse_loss = -torch.mean(torch.sum(
        pos_intensity_norm * torch.log(pos_intensity_norm + NORM_EPS), dim=1,
    ))
    loss_components["sparse"] = sparse_loss

    # 5. Smooth loss - penalize abrupt intensity changes
    if pos_intensity.shape[1] > 1:
        intensity_diff = (pos_intensity[:, 1:] - pos_intensity[:, :-1]).abs()
        smooth_loss = intensity_diff.mean()
    else:
        smooth_loss = torch.tensor(0.0)
    loss_components["smooth"] = smooth_loss

    # 6. Independence loss - directions within manifold
    independence_loss = torch.tensor(0.0)
    for layer in layer_names:
        dirs = direction_params[layer]
        dirs_norm = F.normalize(dirs, p=2, dim=1)
        K = dirs_norm.shape[0]

        if K > 1:
            cos_sim = dirs_norm @ dirs_norm.T
            mask = 1 - torch.eye(K, device=cos_sim.device)

            # Penalize too high or too low similarity
            too_similar = F.relu(cos_sim - self.config.max_cosine_similarity)
            too_different = F.relu(self.config.min_cosine_similarity - cos_sim)
            independence_loss = independence_loss + ((too_similar + too_different) * mask).mean()

    independence_loss = independence_loss / len(layer_names)
    loss_components["independence"] = independence_loss

    # 6. Gate discrimination loss
    # Pos should have high gate (target=1), neg should have low gate (target=0)
    # Use BCE loss which provides gradient even when predictions are at 0.5
    # (The old relu-based loss had zero gradient at 0.5, causing the network to get stuck)
    _upper = STEERING_SCALE_IDENTITY - NORM_EPS
    pos_gate_clamped = pos_gate.clamp(NORM_EPS, _upper)
    neg_gate_clamped = neg_gate.clamp(NORM_EPS, _upper)
    gate_loss = (
        F.binary_cross_entropy(pos_gate_clamped, torch.ones_like(pos_gate))
        + F.binary_cross_entropy(neg_gate_clamped, torch.zeros_like(neg_gate))
        + F.relu(self.config.create_gate_threshold - pos_gate).mean()
        + F.relu(neg_gate - self.config.create_gate_threshold).mean()
    )
    loss_components["gate"] = gate_loss

    # 7. Direction utility loss - reward directions that SEPARATE pos from neg
    # FIXED: Use (pos - neg) not (pos - abs(neg))
    direction_utility_loss = torch.tensor(0.0)
    if direction_weight_params is not None:
        for layer in layer_names:
            dirs = direction_params[layer]  # [K, H]
            dirs_norm = F.normalize(dirs, p=2, dim=1)
            K = dirs_norm.shape[0]

            if K > 1:
                pos_data = data["pos"][layer]  # [N, H]
                neg_data = data["neg"][layer]  # [N, H]

                # Compute per-direction projections
                pos_projs = pos_data @ dirs_norm.T  # [N, K]
                neg_projs = neg_data @ dirs_norm.T  # [N, K]

                # Per-direction utility: how well does this direction SEPARATE pos from neg?
                # FIXED: Use mean(pos) - mean(neg), NOT mean(pos) - mean(abs(neg))
                # Higher value = better separation (pos projects higher than neg)
                dir_utility = pos_projs.mean(dim=0) - neg_projs.mean(dim=0)  # [K]

                # Get current weights
                current_weights = F.softmax(direction_weight_params[layer], dim=0)

                # Weighted utility: reward putting weight on high-utility directions
                # Negate because we want to MAXIMIZE weighted utility (minimize negative)
                weighted_utility = -(current_weights * dir_utility).sum()
                direction_utility_loss = direction_utility_loss + weighted_utility

        direction_utility_loss = direction_utility_loss / len(layer_names)

    loss_components["direction_utility"] = direction_utility_loss

    # 8. Direction weight concentration loss - encourage non-uniform weights
    direction_concentration_loss = torch.tensor(0.0)
    if direction_weight_params is not None:
        for layer in layer_names:
            weights = F.softmax(direction_weight_params[layer], dim=0)
            K = weights.shape[0]
            if K > 1:
                # Negative entropy - encourages sparsity/concentration
                entropy = -(weights * torch.log(weights + NORM_EPS)).sum()
                max_entropy = torch.log(torch.tensor(float(K)))
                normalized_entropy = entropy / max_entropy

                # Also add concentration reward: maximize squared weights
                concentration = -(weights ** 2).sum()

                direction_concentration_loss = (
                    direction_concentration_loss + normalized_entropy + concentration
                )

        direction_concentration_loss = direction_concentration_loss / len(layer_names)

    loss_components["direction_concentration"] = direction_concentration_loss

    # 9. CAA alignment loss - keep primary direction aligned with mean difference
    # This ensures we don't drift away from the empirically-derived truthfulness direction
    caa_alignment_loss = torch.tensor(0.0)
    for layer in layer_names:
        pos_data = data["pos"][layer]
        neg_data = data["neg"][layer]
        dirs = direction_params[layer]

        # Compute CAA direction (mean difference)
        caa_dir = pos_data.mean(dim=0) - neg_data.mean(dim=0)
        caa_dir = F.normalize(caa_dir.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Primary direction (first direction, which was initialized with CAA)
        primary_dir = F.normalize(dirs[0:1], p=2, dim=1).squeeze(0)

        # Cosine similarity - we want it close to 1.0
        cos_sim = (primary_dir * caa_dir).sum()

        # Loss: penalize deviation from CAA direction
        caa_alignment_loss = caa_alignment_loss + (1.0 - cos_sim)

    caa_alignment_loss = caa_alignment_loss / len(layer_names)
    loss_components["caa_alignment"] = caa_alignment_loss

    # Combine losses with warmup
    # IMPORTANT: Contrastive loss is the PRIMARY loss - give it highest weight
    cw = contrastive_weight
    uw = utility_weight
    conc_w = concentration_weight
    caa_w = caa_alignment_weight

    if step < self.config.warmup_steps:
        # Warmup: focus on contrastive + CAA alignment
        total_loss = (
            cw * contrastive_loss +
            self.config.behavior_weight * behavior_loss +
            self.config.retain_weight * retain_loss +
            caa_w * caa_alignment_loss +
            gate_warmup_weight * gate_loss
        )
    else:
        # Full training with all losses
        total_loss = (
            cw * contrastive_loss +
            self.config.behavior_weight * behavior_loss +
            self.config.retain_weight * retain_loss +
            self.config.sparse_weight * sparse_loss +
            self.config.smooth_weight * smooth_loss +
            self.config.independence_weight * independence_loss +
            gate_loss +
            uw * direction_utility_loss +
            conc_w * direction_concentration_loss +
            caa_w * caa_alignment_loss
        )

    return total_loss, loss_components

def _apply_direction_constraints_impl(self, directions: torch.Tensor) -> torch.Tensor:
    """Apply constraints to direction manifold."""
    # Normalize
    directions = F.normalize(directions, p=2, dim=1)

    # Cone constraint: all directions in same half-space as first
    if directions.shape[0] > 1:
        primary = directions[0:1]
        for i in range(1, directions.shape[0]):
            cos_sim = (directions[i:i+1] * primary).sum()
            if cos_sim < 0:
                directions[i] = -directions[i]

    return directions

def _collect_from_set_impl(
    self, pair_set: ContrastivePairSet
) -> Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """Collect configured layers from one shared, complete set of pair rows.

    Every returned layer is appended in ``pair_set`` order from the same pairs.
    Pairs missing any configured sensor, steering, or geometry layer are excluded
    as a unit, so sensor-derived gates/intensities cannot be combined with a
    different steering example. Before layer resolution, all returned layers are
    instead required to have full support across every included pair.
    """
    pair_rows = []
    observed_layers = set()
    for pair in pair_set.pairs:
        pos_la = getattr(pair.positive_response, "layers_activations", None)
        neg_la = getattr(pair.negative_response, "layers_activations", None)
        if pos_la is None or neg_la is None:
            continue

        pos_by_layer = pos_la.to_dict()
        neg_by_layer = neg_la.to_dict()
        supported_layers = {
            layer
            for layer in pos_by_layer.keys() & neg_by_layer.keys()
            if isinstance(pos_by_layer[layer], torch.Tensor)
            and isinstance(neg_by_layer[layer], torch.Tensor)
        }
        pair_rows.append((pos_by_layer, neg_by_layer, supported_layers))
        observed_layers.update(supported_layers)

    if not pair_rows:
        raise InsufficientDataError(reason="No tensor-backed activation pairs found")

    config = getattr(self, "config", None)
    steering_layers = getattr(config, "steering_layers", None)
    sensor_layer = getattr(config, "sensor_layer", None)
    if steering_layers is not None and sensor_layer is not None:
        required_indices = set(steering_layers)
        required_indices.add(sensor_layer)
        geometry_layer = getattr(config, "geometry_analysis_layer", None)
        if geometry_layer is not None:
            required_indices.add(geometry_layer)

        required_layers = set()
        resolved_indices = set()
        for layer in observed_layers:
            try:
                layer_index = int(str(layer).split("_")[-1])
            except (ValueError, IndexError):
                continue
            if layer_index in required_indices:
                required_layers.add(layer)
                resolved_indices.add(layer_index)
        missing_indices = required_indices - resolved_indices
        if missing_indices:
            raise InsufficientDataError(
                reason=(
                    "Missing configured activation layer support for indices "
                    f"{sorted(missing_indices)}"
                )
            )
    else:
        required_layers = set.intersection(
            *(supported for _, _, supported in pair_rows)
        )

    if not required_layers:
        raise InsufficientDataError(
            reason="No configured activation layers have complete pair support"
        )
    complete_rows = [
        (pos_by_layer, neg_by_layer)
        for pos_by_layer, neg_by_layer, supported_layers in pair_rows
        if required_layers <= supported_layers
    ]
    if not complete_rows:
        raise InsufficientDataError(
            reason=(
                "No activation pair has complete sensor, steering, and geometry "
                "layer support"
            )
        )

    buckets: Dict[
        LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]
    ] = {layer: ([], []) for layer in sorted(required_layers, key=str)}
    for pos_by_layer, neg_by_layer in complete_rows:
        for layer, (positive_rows, negative_rows) in buckets.items():
            positive_rows.append(pos_by_layer[layer])
            negative_rows.append(neg_by_layer[layer])

    return buckets

def get_training_logs_impl(self) -> List[Dict[str, Any]]:
    """Return training logs."""
    return self._training_logs
