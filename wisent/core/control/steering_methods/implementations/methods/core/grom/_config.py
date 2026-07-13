"""GROM config and geometry adaptation dataclasses."""
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerName
from wisent.core.control.steering_methods.configs.optimal import get_optimal

# Re-export network classes from _networks for backwards compatibility
from wisent.core.control.steering_methods.methods.grom._networks import (  # noqa: F401
    GatingNetwork,
    IntensityNetwork,
    DirectionWeightNetwork,
)


@dataclass
class GROMConfig:
    """Configuration for GROM steering method."""

    # Required numerical hyperparameters (no defaults -- must come from optimizer)
    num_directions: int
    """Number of directions per layer in the steering manifold."""
    optimization_steps: int
    """Total optimization steps."""
    learning_rate: float
    """Learning rate for all components."""
    warmup_steps: int
    """Steps to warmup (train manifold only before adding networks)."""
    behavior_weight: float
    """Weight for behavior effectiveness loss."""
    retain_weight: float
    """Weight for retain loss (minimize side effects)."""
    sparse_weight: float
    """Weight for sparsity loss (encourage sparse layer activation)."""
    smooth_weight: float
    """Weight for smoothness loss (penalize abrupt intensity changes)."""
    independence_weight: float
    """Weight for direction independence loss."""
    max_alpha: float
    """Maximum steering intensity."""
    gate_temperature: float
    """Temperature for gate sigmoid."""
    min_cosine_similarity: float
    """Minimum cosine similarity between directions."""
    max_cosine_similarity: float
    """Maximum cosine similarity (avoid redundancy)."""
    weight_decay: float
    """Weight decay for AdamW optimizer."""
    max_grad_norm: float
    """Maximum gradient norm for clipping."""
    eta_min_factor: float
    """Factor to compute minimum LR for cosine annealing (eta_min = lr * factor)."""
    linear_threshold: float
    """If linear score > threshold, simplify to single direction."""

    # Geometry adaptation parameters (no defaults -- must come from caller)
    adapt_cone_threshold: float
    """Cone score threshold above which cone-based direction adaptation triggers."""
    adapt_manifold_threshold: float
    """Manifold score threshold above which complex direction adaptation triggers."""
    adapt_linear_directions: int
    """Number of directions to use when linear structure is detected."""
    adapt_complex_directions: int
    """Number of directions for complex (manifold/orthogonal) structures."""
    adapt_max_directions: int
    """Maximum number of directions allowed during geometry adaptation."""
    significant_directions_default: int
    """Value used for significant_directions when cone details lack the key."""
    min_adapted_directions: int
    """Minimum number of directions after geometry adaptation."""
    caa_similarity_skip: float
    """Cosine similarity threshold above which PCA component is skipped."""

    # Loss weights (no defaults -- must come from optimizer or CLI)
    contrastive_margin: float
    """Margin for contrastive loss."""
    contrastive_weight: float
    """Weight for contrastive loss."""
    utility_weight: float
    """Weight for utility loss."""
    concentration_weight: float
    """Weight for concentration loss."""
    gate_warmup_weight: float
    """Weight for gate warmup loss."""
    caa_alignment_weight: float
    """Weight for CAA alignment loss."""
    create_noise_scale: float
    """Noise applied to non-primary manifold direction initialization."""
    create_gate_threshold: float
    """Gate separation threshold used during joint optimization."""

    # Network dimension bounds (no defaults -- must come from caller)
    gate_dim_min: int
    """Minimum gate hidden dimension."""
    gate_dim_max: int
    """Maximum gate hidden dimension."""
    gate_dim_divisor: int
    """Divisor for computing gate hidden dim from model hidden dim."""
    intensity_dim_min: int
    """Minimum intensity hidden dimension."""
    intensity_dim_max: int
    """Maximum intensity hidden dimension."""
    intensity_dim_divisor: int
    """Divisor for computing intensity hidden dim from model hidden dim."""
    gate_shrink_factor: int
    """Internal shrink factor for the gating network's bottleneck layer."""

    # Auto-computed / optional fields (with defaults)
    steering_layers: Optional[List[int]] = None
    """Layer indices where steering can be applied."""
    sensor_layer: Optional[int] = None
    """Primary layer for gating decisions."""
    num_layers: Optional[int] = None
    """Total layers in the model."""

    def resolve_layers(self, num_layers: int) -> None:
        """Resolve steering_layers and sensor_layer based on model's num_layers."""
        self.num_layers = num_layers
        if self.sensor_layer is None:
            raise ValueError(
                "sensor_layer must be specified explicitly. "
                "Pass an integer layer index."
            )
        if self.steering_layers is None:
            raise ValueError(
                "steering_layers must be specified explicitly. "
                "Pass a list of integer layer indices."
            )

    # Network architecture (auto-computed from model hidden_dim)
    gate_hidden_dim: Optional[int] = None
    """Hidden dimension for gating network. If None, auto-computed."""
    intensity_hidden_dim: Optional[int] = None
    """Hidden dimension for intensity network. If None, auto-computed."""
    routing_hidden_dim: Optional[int] = None
    """Hidden dimension for direction routing network. If None, auto-computed."""
    input_dependent_routing: bool = True
    """If True, use a network to predict direction weights per-input."""

    def resolve_network_dims(self, hidden_dim: int) -> None:
        """Resolve and clamp sampled network dimensions for this model width."""
        gate_low, gate_high = sorted((self.gate_dim_min, self.gate_dim_max))
        intensity_low, intensity_high = sorted((
            self.intensity_dim_min, self.intensity_dim_max,
        ))
        gate_candidate = (
            hidden_dim // self.gate_dim_divisor
            if self.gate_hidden_dim is None else self.gate_hidden_dim
        )
        intensity_candidate = (
            hidden_dim // self.intensity_dim_divisor
            if self.intensity_hidden_dim is None else self.intensity_hidden_dim
        )
        self.gate_hidden_dim = max(gate_low, min(gate_high, gate_candidate))
        self.intensity_hidden_dim = max(
            intensity_low, min(intensity_high, intensity_candidate),
        )

    # Boolean flags (sensible defaults)
    use_caa_init: bool = field(default_factory=lambda: get_optimal("use_caa_init"))
    """Initialize primary direction with CAA."""
    normalize: bool = field(default_factory=lambda: get_optimal("normalize"))
    """L2-normalize directions."""
    adapt_to_geometry: bool = True
    """Whether to analyze geometry and adapt configuration."""
    geometry_analysis_layer: Optional[int] = None
    """Layer to use for geometry analysis. If None, uses sensor_layer."""
    max_clusters: int = None
    """Maximum clusters for geometry analysis. Must be set by caller."""
    manifold_neighbors: int = None
    """Number of manifold neighbors for geometry analysis. Must be set by caller."""
    skip_gating_if_linear: bool = True
    """Skip gating network if structure is clearly linear."""
    auto_num_directions: bool = field(default_factory=lambda: get_optimal("auto_num_directions"))
    """Automatically determine num_directions based on geometry."""


@dataclass
class GeometryAdaptation:
    """Results from geometry analysis and adaptations made."""
    detected_structure: str
    """Primary detected structure type (linear, cone, manifold, etc.)."""
    structure_scores: Dict[str, float]
    """Scores for all structure types."""
    adaptations_made: List[str]
    """List of adaptations applied based on geometry."""
    original_num_directions: int
    """Originally configured num_directions."""
    adapted_num_directions: int
    """Num directions after adaptation."""
    gating_enabled: bool
    """Whether gating network is enabled."""
    recommendation: str
    """Steering method recommendation based on geometry."""
