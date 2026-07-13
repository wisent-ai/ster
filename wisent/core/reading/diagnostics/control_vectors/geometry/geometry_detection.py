"""Main geometry structure detection entry point."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Tuple

import torch

from wisent.core.utils.config_tools.constants import (
    GEOMETRY_CONFIDENCE_EXPECTED_PAIR_SUPPORT,
    GEOMETRY_CONFIDENCE_EXPECTED_SAMPLE_SUPPORT,
    GEOMETRY_THRESHOLD_CLUSTER,
    GEOMETRY_THRESHOLD_DEFAULT,
    GEOMETRY_THRESHOLD_MANIFOLD,
    GEOMETRY_THRESHOLD_SPARSE,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
)

from .geometry_types import (
    StructureType,
    StructureScore,
    GeometryAnalysisConfig,
    GeometryAnalysisResult,
)
from .geometry_detectors import (
    detect_linear_structure,
    detect_cone_structure_score,
    detect_cluster_structure,
    detect_manifold_structure,
    detect_sparse_structure,
    detect_bimodal_structure,
    detect_orthogonal_structure,
)

__all__ = ["detect_geometry_structure"]


def detect_geometry_structure(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    config: GeometryAnalysisConfig | None = None,
    geometry_threshold_default: float = GEOMETRY_THRESHOLD_DEFAULT,
    geometry_threshold_cluster: float = GEOMETRY_THRESHOLD_CLUSTER,
    geometry_threshold_sparse: float = GEOMETRY_THRESHOLD_SPARSE,
    geometry_threshold_manifold: float = GEOMETRY_THRESHOLD_MANIFOLD,
) -> GeometryAnalysisResult:
    """Detect the geometric structure of activation differences."""
    geometry_threshold_default = _validate_threshold(
        "geometry_threshold_default", geometry_threshold_default,
    )
    geometry_threshold_cluster = _validate_threshold(
        "geometry_threshold_cluster", geometry_threshold_cluster,
    )
    geometry_threshold_sparse = _validate_threshold(
        "geometry_threshold_sparse", geometry_threshold_sparse,
    )
    geometry_threshold_manifold = _validate_threshold(
        "geometry_threshold_manifold", geometry_threshold_manifold,
    )
    cfg = config or GeometryAnalysisConfig()

    pos_tensor = pos_activations.detach().float()
    neg_tensor = neg_activations.detach().float()

    if pos_tensor.dim() == 1:
        pos_tensor = pos_tensor.unsqueeze(0)
    if neg_tensor.dim() == 1:
        neg_tensor = neg_tensor.unsqueeze(0)

    n_pairs = min(pos_tensor.shape[0], neg_tensor.shape[0])
    diff_vectors = pos_tensor[:n_pairs] - neg_tensor[:n_pairs]

    raw_scores: Dict[str, StructureScore] = {}
    support_confidence = min(
        SCORE_RANGE_MAX,
        n_pairs / GEOMETRY_CONFIDENCE_EXPECTED_PAIR_SUPPORT,
    )
    raw_scores["linear"] = detect_linear_structure(
        pos_tensor, neg_tensor, diff_vectors, cfg,
        detector_cohens_d_divisor=float(max(1, cfg.num_components)),
        detector_large_sample_n=GEOMETRY_CONFIDENCE_EXPECTED_SAMPLE_SUPPORT,
    )
    raw_scores["cone"] = detect_cone_structure_score(
        pos_tensor, neg_tensor, cfg,
        detector_small_sample_n=GEOMETRY_CONFIDENCE_EXPECTED_PAIR_SUPPORT,
    )
    raw_scores["cluster"] = detect_cluster_structure(
        pos_tensor, neg_tensor, diff_vectors, cfg,
        min_clusters=cfg.min_clusters,
        detector_cluster_sample_n=GEOMETRY_CONFIDENCE_EXPECTED_SAMPLE_SUPPORT,
    )
    raw_scores["manifold"] = detect_manifold_structure(
        pos_tensor, neg_tensor, diff_vectors, cfg,
        geo_manifold_score_default=SCORE_RANGE_MIN,
        geo_manifold_confidence=support_confidence,
    )
    raw_scores["sparse"] = detect_sparse_structure(
        pos_tensor, neg_tensor, diff_vectors, cfg,
        geo_diag_sparse_threshold=geometry_threshold_sparse,
        sparse_detection_confidence=support_confidence,
    )
    raw_scores["bimodal"] = detect_bimodal_structure(
        pos_tensor, neg_tensor, diff_vectors, cfg,
        bimodal_detection_confidence=support_confidence,
    )
    raw_scores["orthogonal"] = detect_orthogonal_structure(
        pos_tensor, neg_tensor, diff_vectors, cfg,
        geo_diag_orthogonal_threshold=geometry_threshold_default,
        geo_orthogonal_confidence=support_confidence,
    )

    best_structure, best_score = _find_most_specific_structure(raw_scores)
    recommendation = _generate_recommendation(best_structure, raw_scores)

    return GeometryAnalysisResult(
        best_structure=best_structure,
        best_score=best_score,
        all_scores=raw_scores,
        recommendation=recommendation,
        details={
            "config": cfg.__dict__,
            "n_positive": pos_tensor.shape[0],
            "n_negative": neg_tensor.shape[0],
            "hidden_dim": pos_tensor.shape[1],
        }
    )


def _find_most_specific_structure(
    scores: Dict[str, StructureScore],
) -> Tuple[StructureType, float]:
    """Return the structure with the highest score. No priority ordering."""
    best_key = max(scores.keys(), key=lambda key: scores[key].score)
    return scores[best_key].structure_type, scores[best_key].score


def _validate_threshold(name: str, value: float) -> float:
    """Validate and normalize a geometry threshold override."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or not SCORE_RANGE_MIN <= normalized <= SCORE_RANGE_MAX:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return normalized


def _generate_recommendation(best_structure: StructureType, all_scores: Dict[str, StructureScore]) -> str:
    """No unvalidated recommendations. Callers should use raw scores in all_scores."""
    return ""
