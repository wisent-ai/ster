"""Activation space geometry types derived from empirical clustering."""
from __future__ import annotations

import json
import math
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET, SCORE_RANGE_MIN, SCORE_RANGE_MAX,
)


class GeometryType(Enum):
    """Coarse activation space geometry types."""
    LINEAR_FLAT = "linear_flat"
    LINEAR_MULTI = "linear_multi"
    CURVED_DIFFUSE = "curved_diffuse"
    UNSTABLE_CONCENTRATED = "unstable_concentrated"
    CONCENTRATED_FEW = "concentrated_few"
    RAW = "raw"


class GeometryTypeFine(Enum):
    """Fine-grained geometry types mapped to A-H profiles."""
    LINEAR_STABLE = "linear_stable"
    DIFFUSE_CURVED = "diffuse_curved"
    CONCENTRATED_FEW = "concentrated_few"
    LINEAR_CONCENTRATED = "linear_concentrated"
    MID_MULTI_CONCEPT = "mid_multi_concept"
    LINEAR_MULTIDIR = "linear_multidir"
    LOW_LINEAR_UNSTABLE = "low_linear_unstable"
    MANY_CONCEPT_CURVED = "many_concept_curved"
    SPHERICAL_NORMALIZED = "spherical_normalized"
    RAW = "raw"


SHAPE_MAP = {
    GeometryTypeFine.LINEAR_STABLE: "linear",
    GeometryTypeFine.LINEAR_CONCENTRATED: "cone",
    GeometryTypeFine.LINEAR_MULTIDIR: "orthogonal",
    GeometryTypeFine.CONCENTRATED_FEW: "bimodal",
    GeometryTypeFine.MID_MULTI_CONCEPT: "cluster",
    GeometryTypeFine.DIFFUSE_CURVED: "manifold",
    GeometryTypeFine.LOW_LINEAR_UNSTABLE: "sparse",
    GeometryTypeFine.MANY_CONCEPT_CURVED: "manifold",
    GeometryTypeFine.SPHERICAL_NORMALIZED: "sphere",
}

PROFILE_METRICS = [
    "signal_strength", "linear_probe_accuracy", "cloud_separation_ratio",
    "manifold_curvature", "concept_coherence", "icd_top1_variance",
    "fisher_fisher_gini", "knn_accuracy",
]

_METRIC_ALIASES = {
    "linear": ["linear_probe_accuracy", "linear_probe"],
    "stability": ["direction_stability_score", "direction_stability"],
    "n_concepts": ["n_concepts"],
    "var_pc1": ["manifold_variance_pc1"],
    "curvature": ["manifold_curvature", "manifold_curvature_proxy"],
    "coherence": ["concept_coherence"],
}

_centroids_cache = None
_fine_type_lookup = {t.value: t for t in GeometryTypeFine}
_coarse_type_lookup = {t.value: t for t in GeometryType}


def _load_centroids():
    """Load profile centroids from bundled JSON."""
    global _centroids_cache
    if _centroids_cache is not None:
        return _centroids_cache
    import wisent
    pkg = Path(wisent.__file__).parent
    path = pkg / "support" / "parameters" / "lm_eval" / "profiles" / "profile_centroids.json"
    with open(path) as f:
        _centroids_cache = json.load(f)
    return _centroids_cache


def _normalize_value(val, rng):
    """Min-max normalize a single value using range dict with min/max keys."""
    mn, mx = rng["min"], rng["max"]
    denom = mx - mn
    if denom <= SCORE_RANGE_MIN:
        return SCORE_RANGE_MIN
    return (val - mn) / denom


def _squared_diff(a, b):
    """Compute squared difference without inline numeric literal."""
    diff = a - b
    return diff * diff


def _extract(metrics: Dict[str, Any], key: str) -> Optional[float]:
    for alias in _METRIC_ALIASES[key]:
        if alias in metrics and metrics[alias] is not None:
            return float(metrics[alias])
    return None


def _classify_fine(metrics: Dict[str, Any]) -> Tuple[GeometryTypeFine, float]:
    """Classify into fine type using nearest-centroid on profile metrics."""
    data = _load_centroids()
    ranges = data["ranges"]
    vals = []
    for key in PROFILE_METRICS:
        v = metrics.get(key)
        if v is None or not isinstance(v, (int, float)):
            return GeometryTypeFine.RAW, SCORE_RANGE_MIN
        vals.append(float(v))
    norm = [_normalize_value(vals[i], ranges[k]) for i, k in enumerate(PROFILE_METRICS)]
    best_profile, best_dist = None, float("inf")
    for profile_name, centroid in data["centroids"].items():
        c_norm = [_normalize_value(centroid[j], ranges[k]) for j, k in enumerate(PROFILE_METRICS)]
        dist = math.sqrt(sum(_squared_diff(a, b) for a, b in zip(norm, c_norm)))
        if dist < best_dist:
            best_dist = dist
            best_profile = profile_name
    fine_value = data["profile_to_fine_type"].get(best_profile, "raw")
    fine_type = _fine_type_lookup.get(fine_value, GeometryTypeFine.RAW)
    max_possible = math.sqrt(len(PROFILE_METRICS))
    confidence = max(SCORE_RANGE_MIN, SCORE_RANGE_MAX - best_dist / max_possible)
    return fine_type, confidence


def _fine_to_coarse(fine_type: GeometryTypeFine) -> GeometryType:
    """Map fine type to coarse type using profile data."""
    data = _load_centroids()
    coarse_value = data["fine_to_coarse"].get(fine_type.value, "raw")
    return _coarse_type_lookup.get(coarse_value, GeometryType.RAW)


def _classify(metrics, centroids, default_type, *, default_score, blend_default,
              default_scale, zwiad_ranges, zwiad_weights):
    """Classify geometry using nearest-centroid distance to profile centroids."""
    fine_type, confidence = _classify_fine(metrics)
    if isinstance(default_type, GeometryType):
        return _fine_to_coarse(fine_type), confidence
    return fine_type, confidence


def classify_geometry(
    metrics: Dict[str, Any], fine: bool = False, *, default_score: float, blend_default: float,
    default_scale: float, zwiad_ranges: Dict[str, float], zwiad_weights: Dict[str, float],
) -> Tuple:
    """Classify geometry. fine=True uses fine types, False uses coarse."""
    kw = dict(default_score=default_score, blend_default=blend_default,
              default_scale=default_scale, zwiad_ranges=zwiad_ranges, zwiad_weights=zwiad_weights)
    if fine:
        return _classify(metrics, None, GeometryTypeFine.LINEAR_STABLE, **kw)
    return _classify(metrics, None, GeometryType.LINEAR_FLAT, **kw)


def _build_categorized_reverse_map():
    """Build {category_task: task} from working_benchmarks_categorized.json."""
    import wisent
    pkg_root = Path(wisent.__file__).parent
    candidates = [
        pkg_root / "support" / "parameters" / "lm_eval" / "working_benchmarks_categorized.json",
    ]
    for c in candidates:
        if c.exists():
            data = json.loads(c.read_text())
            rmap = {}
            for category, tasks in data.items():
                for task in tasks:
                    rmap[f"{category}_{task}"] = task
            return rmap
    return {}


def select_representative_benchmarks(
    zwiad_dir: str, model_slug: str, *, per_type: int, fine: bool = False,
    default_score: float, blend_default: float, default_scale: float,
    zwiad_ranges: Dict[str, float], zwiad_weights: Dict[str, float],
) -> Dict:
    """Select representative benchmarks covering all types."""
    zwiad_path = Path(zwiad_dir)
    prefix = model_slug.replace("/", "_")
    enum_cls = GeometryTypeFine if fine else GeometryType
    rev_map = _build_categorized_reverse_map()
    typed = {t: [] for t in enum_cls}
    geo_kw = dict(default_score=default_score, blend_default=blend_default,
                  default_scale=default_scale, zwiad_ranges=zwiad_ranges, zwiad_weights=zwiad_weights)
    for f in sorted(zwiad_path.glob(f"{prefix}__*.json")):
        bench = f.stem.replace(f"{prefix}__", "")
        task_name = rev_map.get(bench, bench)
        data = json.loads(f.read_text())
        m = data.get("metrics", data)
        gtype, conf = classify_geometry(m, fine=fine, **geo_kw)
        typed[gtype].append((task_name, conf))
    selected = {}
    for gtype in enum_cls:
        cands = typed[gtype]
        cands.sort(key=lambda x: x[COMBO_OFFSET], reverse=True)
        selected[gtype] = [b for b, _ in cands[:per_type]]
    return selected


@dataclass
class GeometryProfile:
    """Full geometry characterization of a benchmark."""
    geometry_type: str
    geometry_type_fine: str
    activation_shape: str
    recommended_methods: List[str]
    confidence: float
    confidence_fine: float
    metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "geometry_type": self.geometry_type,
            "geometry_type_fine": self.geometry_type_fine,
            "activation_shape": self.activation_shape,
            "recommended_methods": self.recommended_methods,
            "confidence": self.confidence,
            "confidence_fine": self.confidence_fine,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GeometryProfile:
        return cls(**d)


def profile_benchmark(
    metrics: Dict[str, Any], *, default_score: float, blend_default: float,
    default_scale: float, zwiad_ranges: Dict[str, float], zwiad_weights: Dict[str, float],
) -> GeometryProfile:
    """Build full geometry profile with both granularities."""
    kw = dict(default_score=default_score, blend_default=blend_default,
              default_scale=default_scale, zwiad_ranges=zwiad_ranges, zwiad_weights=zwiad_weights)
    gt5, c5 = classify_geometry(metrics, fine=False, **kw)
    gt8, c8 = classify_geometry(metrics, fine=True, **kw)
    key_metrics = {}
    for key in _METRIC_ALIASES:
        v = _extract(metrics, key)
        if v is not None:
            key_metrics[key] = v
    for key in PROFILE_METRICS:
        v = metrics.get(key)
        if v is not None and isinstance(v, (int, float)):
            key_metrics[key] = float(v)
    return GeometryProfile(
        geometry_type=gt5.value,
        geometry_type_fine=gt8.value,
        activation_shape=SHAPE_MAP.get(gt8, "raw"),
        recommended_methods=[],
        confidence=c5,
        confidence_fine=c8,
        metrics=key_metrics,
    )
