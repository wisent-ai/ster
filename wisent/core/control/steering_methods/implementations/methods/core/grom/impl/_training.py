"""GROM training helpers: geometry analysis, direction init, data prep."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.config_tools.constants import COMBO_OFFSET, RECURSION_INITIAL_DEPTH  # noqa: E501
from wisent.core.control.steering_methods.methods.grom._config import GeometryAdaptation
from wisent.core.utils.infra_tools.errors import InsufficientDataError

def _validate_aligned_support(
    buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
    layer_names: List[LayerName],
) -> int:
    """Require equal positive/negative pair support across all selected layers.

    ``_collect_from_set_impl`` builds each layer from the same pair rows in the
    same order. Equal support here both enforces that collector contract and
    rejects manually supplied/incomplete buckets before tensor arithmetic.
    """
    if not layer_names:
        raise InsufficientDataError(reason="No activation layers selected")

    expected_support = None
    for layer in layer_names:
        if layer not in buckets:
            raise InsufficientDataError(reason=f"Missing activation layer {layer!r}")
        pos_rows, neg_rows = buckets[layer]
        if len(pos_rows) != len(neg_rows):
            raise InsufficientDataError(
                reason=(
                    f"Layer {layer!r} has unpaired activation support: "
                    f"{len(pos_rows)} positive rows and {len(neg_rows)} negative rows"
                )
            )
        if not pos_rows:
            raise InsufficientDataError(
                reason=f"Layer {layer!r} has no complete activation pairs"
            )
        if expected_support is None:
            expected_support = len(pos_rows)
        elif len(pos_rows) != expected_support:
            raise InsufficientDataError(
                reason=(
                    "Sensor and steering layers do not share identical full pair "
                    f"support: expected {expected_support} rows, but layer "
                    f"{layer!r} has {len(pos_rows)}"
                )
            )

    return expected_support


def _analyze_and_adapt_geometry_impl(
    self,
    buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
    layer_names: List[LayerName],
    hidden_dim: int,
    default_score: float,
) -> GeometryAdaptation:
    """
    Analyze geometry of activations and adapt GROM configuration.

    Returns:
        GeometryAdaptation with detected structure and adaptations made.
    """
    from wisent.core.primitives.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_structure,
        GeometryAnalysisConfig,
    )

    analysis_layer_idx = (
        self.config.geometry_analysis_layer
        if self.config.geometry_analysis_layer is not None
        else self.config.sensor_layer
    )
    analysis_layer = None
    for layer in buckets:
        try:
            layer_index = int(str(layer).split("_")[-1])
        except (ValueError, IndexError):
            continue
        if layer_index == analysis_layer_idx:
            if analysis_layer is not None:
                raise InsufficientDataError(
                    reason=f"Multiple activation names resolve to geometry layer {layer_index}"
                )
            analysis_layer = layer
    if analysis_layer is None:
        raise InsufficientDataError(
            reason=f"Missing configured geometry layer {analysis_layer_idx}"
        )

    pos_list, neg_list = buckets[analysis_layer]
    pair_support = _validate_aligned_support(buckets, [analysis_layer])
    pos_tensor = torch.stack(
        [tensor.detach().float().reshape(-1) for tensor in pos_list], dim=0,
    )
    neg_tensor = torch.stack(
        [tensor.detach().float().reshape(-1) for tensor in neg_list], dim=0,
    )

    combined_support = pair_support * 2
    min_clusters = 2
    max_supported_clusters = max(min_clusters, combined_support // 2)
    configured_max_clusters = self.config.max_clusters
    if configured_max_clusters is None:
        max_clusters = max(
            min_clusters,
            min(self.config.num_directions, max_supported_clusters),
        )
    else:
        if type(configured_max_clusters) is not int:
            raise ValueError(
                f"max_clusters must be an integer; got {configured_max_clusters!r}"
            )
        max_clusters = min(
            max(configured_max_clusters, min_clusters), max_supported_clusters,
        )
    max_supported_neighbors = max(1, pair_support - 1)
    configured_neighbors = self.config.manifold_neighbors
    if configured_neighbors is None:
        manifold_neighbors = max(
            1, min(self.config.num_directions, max_supported_neighbors),
        )
    else:
        if type(configured_neighbors) is not int:
            raise ValueError(
                "manifold_neighbors must be an integer; "
                f"got {configured_neighbors!r}"
            )
        manifold_neighbors = min(
            max(configured_neighbors, 1), max_supported_neighbors,
        )

    geo_config = GeometryAnalysisConfig(
        num_components=max(
            1, min(self.config.num_directions, hidden_dim, pair_support),
        ),
        optimization_steps=max(1, self.config.optimization_steps),
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        manifold_neighbors=manifold_neighbors,
        kmeans_max_iterations=max(1, self.config.optimization_steps),
    )
    geo_result = detect_geometry_structure(
        pos_tensor,
        neg_tensor,
        geo_config,
        geometry_threshold_default=self.config.linear_threshold,
        geometry_threshold_cluster=self.config.adapt_cone_threshold,
        geometry_threshold_sparse=self.config.adapt_cone_threshold,
        geometry_threshold_manifold=self.config.adapt_manifold_threshold,
    )

    # Extract scores
    structure_scores = {
        name: score.score for name, score in geo_result.all_scores.items()
    }
    detected_structure = geo_result.best_structure.value

    # Determine adaptations
    adaptations = []
    original_num_directions = self.config.num_directions
    adapted_num_directions = original_num_directions
    gating_enabled = True

    linear_score = structure_scores.get("linear", default_score)
    cone_score = structure_scores.get("cone", default_score)
    manifold_score = structure_scores.get("manifold", default_score)

    # Simplify if linear
    if linear_score > self.config.linear_threshold:
        if self.config.auto_num_directions:
            adapted_num_directions = self.config.adapt_linear_directions
            adaptations.append(f"Reduced num_directions to {self.config.adapt_linear_directions} (linear score={linear_score:.2f})")

        if self.config.skip_gating_if_linear:
            gating_enabled = False
            adaptations.append("Disabled gating network (linear structure)")

    # Adjust directions based on cone structure
    elif cone_score > self.config.adapt_cone_threshold and self.config.auto_num_directions:
        # Cone structure benefits from multiple directions
        cone_details = geo_result.all_scores.get("cone")
        if cone_details and hasattr(cone_details, "details"):
            sig_dirs = cone_details.details.get(
                "significant_directions", self.config.significant_directions_default,
            )
            adapted_num_directions = max(
                self.config.min_adapted_directions,
                min(sig_dirs + COMBO_OFFSET, self.config.adapt_max_directions),
            )
            if adapted_num_directions != original_num_directions:
                adaptations.append(
                    f"Adjusted num_directions to {adapted_num_directions} based on cone structure"
                )

    # Increase directions for complex structures
    elif (
        manifold_score > self.config.adapt_manifold_threshold
        or structure_scores.get("orthogonal", default_score) > self.config.adapt_cone_threshold
    ):
        if self.config.auto_num_directions and adapted_num_directions < self.config.adapt_complex_directions:
            adapted_num_directions = self.config.adapt_complex_directions
            adaptations.append(
                f"Increased num_directions to {self.config.adapt_complex_directions} for manifold/orthogonal structure"
            )

    if not adaptations:
        adaptations.append("No adaptations needed - using default configuration")

    return GeometryAdaptation(
        detected_structure=detected_structure,
        structure_scores=structure_scores,
        adaptations_made=adaptations,
        original_num_directions=original_num_directions,
        adapted_num_directions=adapted_num_directions,
        gating_enabled=gating_enabled,
        recommendation=geo_result.recommendation,
    )

def _initialize_directions_impl(
    self,
    buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
    layer_names: List[LayerName],
    hidden_dim: int,
    num_directions: Optional[int] = None,
) -> Dict[LayerName, torch.Tensor]:
    """Initialize direction manifold for each layer using PCA for diversity."""
    directions = {}
    K = num_directions if num_directions is not None else self.config.num_directions
    _validate_aligned_support(buckets, layer_names)

    for layer in layer_names:
        pos_list, neg_list = buckets[layer]

        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

        # First direction: CAA (mean difference)
        caa_dir = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        caa_dir = F.normalize(caa_dir, p=2, dim=0)

        # Use PCA on difference vectors for diverse directions
        diff_vectors = pos_tensor - neg_tensor  # [N, H]

        # Center the difference vectors
        diff_centered = diff_vectors - diff_vectors.mean(dim=0, keepdim=True)

        # SVD for PCA
        try:
            # Use svd_lowrank for efficiency on large hidden_dim
            U, S, V = torch.svd_lowrank(diff_centered, q=min(K + 2, diff_centered.shape[0], diff_centered.shape[1]))
            # V contains principal components: [H, K+2]
            pca_dirs = V.T  # [K+2, H]
        except Exception:
            # Use random orthogonal initialization if SVD fails
            pca_dirs = None

        dirs = torch.zeros(K, hidden_dim)
        dirs[0] = caa_dir  # First direction is always CAA

        if pca_dirs is not None and K > 1:
            # Use PCA directions for remaining directions
            # Skip the first PCA component if it's too similar to CAA
            pca_idx = 0
            for i in range(1, K):
                if pca_idx < pca_dirs.shape[0]:
                    candidate = F.normalize(pca_dirs[pca_idx], p=2, dim=0)
                    cos_with_caa = (candidate * caa_dir).sum().abs()

                    # If too similar to CAA, try next PCA component
                    next_idx = pca_idx + COMBO_OFFSET
                    if cos_with_caa > self.config.caa_similarity_skip and next_idx < pca_dirs.shape[RECURSION_INITIAL_DEPTH]:
                        pca_idx = next_idx
                        candidate = F.normalize(pca_dirs[pca_idx], p=2, dim=0)

                    dirs[i] = F.normalize(
                        candidate + torch.randn_like(candidate) * self.config.create_noise_scale,
                        p=2,
                        dim=0,
                    )
                    pca_idx += 1
                else:
                    # Orthogonalize random vector against previous directions
                    random_dir = torch.randn(hidden_dim)
                    # Gram-Schmidt orthogonalization
                    for j in range(i):
                        proj = (random_dir * dirs[j]).sum() * dirs[j]
                        random_dir = random_dir - proj
                    dirs[i] = F.normalize(
                        random_dir + torch.randn_like(random_dir) * self.config.create_noise_scale,
                        p=2,
                        dim=0,
                    )
        else:
            # Use Gram-Schmidt orthogonalization for all directions
            for i in range(1, K):
                random_dir = torch.randn(hidden_dim)
                for j in range(i):
                    proj = (random_dir * dirs[j]).sum() * dirs[j]
                    random_dir = random_dir - proj
                dirs[i] = F.normalize(
                    random_dir + torch.randn_like(random_dir) * self.config.create_noise_scale,
                    p=2,
                    dim=0,
                )

        # Ensure all in same half-space as CAA
        for i in range(1, K):
            if (dirs[i] * caa_dir).sum() < 0:
                dirs[i] = -dirs[i]

        directions[layer] = dirs

    return directions

def _prepare_data_tensors_impl(
    self,
    buckets: Dict[LayerName, Tuple[List[torch.Tensor], List[torch.Tensor]]],
    layer_names: List[LayerName],
) -> Dict[str, Dict[LayerName, torch.Tensor]]:
    """Prepare stacked tensors for training."""
    _validate_aligned_support(buckets, layer_names)
    data = {"pos": {}, "neg": {}}

    for layer in layer_names:
        pos_list, neg_list = buckets[layer]
        data["pos"][layer] = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        data["neg"][layer] = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

    return data

def _find_sensor_layer_impl(self, layer_names: List[LayerName]) -> LayerName:
    """Resolve the exact configured sensor layer from available names."""
    matches = []
    for layer in layer_names:
        try:
            layer_index = int(str(layer).split("_")[-1])
        except (ValueError, IndexError):
            continue
        if layer_index == self.config.sensor_layer:
            matches.append(layer)
    if len(matches) != 1:
        raise InsufficientDataError(
            reason=(
                f"Expected exactly one configured sensor layer {self.config.sensor_layer}; "
                f"found {len(matches)}"
            )
        )
    return matches[0]
