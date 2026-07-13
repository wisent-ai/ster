"""Geometry detection type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Tuple

from wisent.core.utils.config_tools.constants import (
    GEOMETRY_DEFAULT_KMEANS_MAX_ITERATIONS,
    GEOMETRY_DEFAULT_MANIFOLD_NEIGHBORS,
    GEOMETRY_DEFAULT_MAX_CLUSTERS,
    GEOMETRY_DEFAULT_MIN_CLUSTERS,
    GEOMETRY_DEFAULT_NUM_COMPONENTS,
    GEOMETRY_DEFAULT_OPTIMIZATION_STEPS,
)


__all__ = [
    "StructureType",
    "StructureScore",
    "GeometryAnalysisConfig",
    "GeometryAnalysisResult",
]


class StructureType(Enum):
    """Types of geometric structures in activation space."""
    LINEAR = "linear"
    CONE = "cone"
    CLUSTER = "cluster"
    MANIFOLD = "manifold"
    SPARSE = "sparse"
    BIMODAL = "bimodal"
    ORTHOGONAL = "orthogonal"
    UNKNOWN = "unknown"


@dataclass
class StructureScore:
    """Score for a single structure type."""
    structure_type: StructureType
    score: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryAnalysisConfig:
    """Configuration for comprehensive geometry analysis."""

    num_components: int = GEOMETRY_DEFAULT_NUM_COMPONENTS
    """Number of components/directions to analyze."""

    optimization_steps: int = GEOMETRY_DEFAULT_OPTIMIZATION_STEPS
    """Steps for optimization-based methods."""

    min_clusters: int = GEOMETRY_DEFAULT_MIN_CLUSTERS
    """Minimum number of clusters."""

    max_clusters: int = GEOMETRY_DEFAULT_MAX_CLUSTERS
    """Maximum number of clusters to try."""

    kmeans_max_iterations: int = GEOMETRY_DEFAULT_KMEANS_MAX_ITERATIONS
    """Maximum iterations for k-means clustering."""

    manifold_neighbors: int = GEOMETRY_DEFAULT_MANIFOLD_NEIGHBORS
    """Number of neighbors for manifold analysis."""

    def __post_init__(self):
        """Validate required fields."""
        for _n, _v in [("num_components", self.num_components), ("optimization_steps", self.optimization_steps), ("min_clusters", self.min_clusters), ("max_clusters", self.max_clusters), ("manifold_neighbors", self.manifold_neighbors), ("kmeans_max_iterations", self.kmeans_max_iterations)]:
            if _v is None: raise ValueError(f"{_n} is required in GeometryAnalysisConfig")

    use_universal_thresholds: bool = True
    """Whether to use thresholds tuned for universal subspace theory."""


@dataclass
class GeometryAnalysisResult:
    """Results from comprehensive geometry analysis."""

    best_structure: StructureType
    """The structure type that best fits the data."""

    best_score: float
    """Score of the best-fitting structure."""

    all_scores: Dict[str, StructureScore]
    """Scores for all analyzed structure types."""

    recommendation: str
    """Recommended steering method based on geometry."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional analysis details."""

    def get_ranking(self) -> List[Tuple[StructureType, float]]:
        """Get structures ranked by score."""
        return sorted(
            [(s.structure_type, s.score) for s in self.all_scores.values()],
            key=lambda x: x[1],
            reverse=True
        )
