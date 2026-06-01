"""Zwiad protocol: representation discovery and characterization."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from wisent.core import constants as _C
from wisent.core.utils.config_tools.constants import (
    ROUNDING_PRECISION, STAT_ALPHA, DIAGNOSTICS_TOTAL_CHECKS,
    LINEARITY_GAP_THRESHOLD, LINEARITY_P_THRESHOLD, LINEARITY_RESIDUAL_THRESHOLD,
    LINEARITY_RAMSEY_THRESHOLD, LINEARITY_N_BOOTSTRAP, LINEARITY_CV_FOLDS,
    LINEARITY_CONFIDENCE_HIGH, LINEARITY_CONFIDENCE_LOW,
    LINEARITY_CROSS_CONTEXT_THRESHOLD,
)
from .zwiad_config import adaptive_gap_threshold, adaptive_min_silhouette, ZwiadProtocolConfig
from .unsupervised.topology import test_topology as _test_topology, result_to_dict as _topology_to_dict

@dataclass
class SignalTestResult:
    """Result of signal existence test."""
    max_z_score: float
    min_p_value: float
    passed: bool
    permutation_metrics: Dict[str, Dict[str, float]]
    nonsense_metrics: Optional[Dict[str, Dict[str, float]]] = None

@dataclass
class GeometryTestResult:
    """Result of geometry (linear vs nonlinear) test."""
    linear_accuracy: float
    nonlinear_accuracy: float
    gap: float
    diagnosis: str
    confidence: float = None
    p_value: Optional[float] = None
    gap_ci_lower: float = None
    gap_ci_upper: float = None
    n_diagnostics_passed: int = 0
    n_diagnostics_total: int = 0
    t_statistic: float = None
    residual_silhouette: float = None
    residuals_cluster: bool = False
    ramsey_improvement: float = None
    ramsey_significant: bool = False
    diagnostics: Optional[Dict[str, Any]] = None

@dataclass
class DecompositionTestResult:
    """Result of concept decomposition test."""
    n_concepts: int
    cluster_labels: List[int]
    silhouette_score: float
    is_fragmented: bool
    per_concept_sizes: Dict[int, int]

@dataclass
class InterventionResult:
    """Result of intervention method selection."""
    recommended_method: str
    confidence: float
    reasoning: List[str]
    method_scores: Dict[str, float]

@dataclass
class EditabilityTestResult:
    """Result of EOT editability test (Step 5)."""
    composite_editability: float
    steering_survival: float
    spectral_concentration: float
    spectral_sharpness: float
    attention_entropy: float
    jacobian_sensitivity: float

def test_signal(
    pos: torch.Tensor, neg: torch.Tensor, metric_keys: List[str],
    device: str,
    model=None, tokenizer=None, layer: int = None,
    p_threshold: float = STAT_ALPHA,
    mlp_early_stopping_min_samples: int = None,
    mlp_probe_max_iter: int = None,
) -> SignalTestResult:
    """Step one: Test if a learnable signal exists relative to null."""
    from ..validation.null_tests.signal_null_tests import compute_signal_vs_null, compute_signal_vs_nonsense, compute_aggregate_signal
    perm_metrics = compute_signal_vs_null(
        pos, neg, metric_keys,
        mlp_early_stopping_min_samples=mlp_early_stopping_min_samples,
        mlp_probe_max_iter=mlp_probe_max_iter)
    max_z, min_p, any_sig = compute_aggregate_signal(perm_metrics, correction="bonferroni")
    nonsense_metrics = None
    if model is not None and tokenizer is not None:
        try:
            nonsense_metrics = compute_signal_vs_nonsense(
                pos, neg, model, tokenizer, metric_keys, device=device, layer=layer,
                mlp_early_stopping_min_samples=mlp_early_stopping_min_samples,
                mlp_probe_max_iter=mlp_probe_max_iter)
            nonsense_z, _, nonsense_sig = compute_aggregate_signal(nonsense_metrics, correction="bonferroni")
            max_z = max(max_z, nonsense_z)
            passed = (min_p < p_threshold) and nonsense_sig
        except Exception:
            passed = min_p < p_threshold
    else:
        passed = min_p < p_threshold
    return SignalTestResult(max_z, min_p, passed, perm_metrics, nonsense_metrics)


def test_geometry(pos: torch.Tensor, neg: torch.Tensor, diagnostics_total_checks: int) -> GeometryTestResult:
    """Step two: Test if geometry is linear or nonlinear."""
    from ..analysis.is_linear import test_linearity
    r = test_linearity(
        pos, neg, diagnostics_total_checks=diagnostics_total_checks,
        cv_folds=LINEARITY_CV_FOLDS, gap_threshold=LINEARITY_GAP_THRESHOLD,
        p_threshold=LINEARITY_P_THRESHOLD,
        residual_threshold=LINEARITY_RESIDUAL_THRESHOLD,
        ramsey_threshold=LINEARITY_RAMSEY_THRESHOLD,
        n_bootstrap=LINEARITY_N_BOOTSTRAP,
        linearity_confidence_high=LINEARITY_CONFIDENCE_HIGH,
        linearity_confidence_low=LINEARITY_CONFIDENCE_LOW,
        linearity_cross_context_threshold=LINEARITY_CROSS_CONTEXT_THRESHOLD,
    )
    return GeometryTestResult(
        linear_accuracy=r.linear_accuracy, nonlinear_accuracy=r.nonlinear_accuracy,
        gap=r.gap, diagnosis=r.diagnosis, confidence=r.confidence, p_value=r.p_value,
        gap_ci_lower=r.gap_ci_lower, gap_ci_upper=r.gap_ci_upper,
        n_diagnostics_passed=r.n_diagnostics_passed, n_diagnostics_total=r.n_diagnostics_total,
        t_statistic=r.t_statistic, residual_silhouette=r.residual_silhouette,
        residuals_cluster=r.residuals_cluster, ramsey_improvement=r.ramsey_improvement,
        ramsey_significant=r.ramsey_significant, diagnostics=r.diagnostics,
    )

def test_decomposition(
    pos: torch.Tensor, neg: torch.Tensor, min_silhouette: float,
    *, decomp_pca_dims_max: int, decomp_max_silhouette_samples: int,
    decomp_adaptive_k_min: int, decomp_adaptive_k_max: int,
    decomp_min_cluster_size_base: int, decomp_cluster_size_ratio: int,
    decomp_kmeans_n_init_min: int, decomp_kmeans_n_init_max: int,
    decomp_kmeans_scaling_factor: int,
) -> DecompositionTestResult:
    """Step three: Test if concept is fragmented into sub-concepts."""
    from ..metrics.distribution.decomposition_metrics import find_optimal_clustering
    diff = pos - neg
    n_concepts, labels, sil = find_optimal_clustering(
        diff, decomp_min_silhouette=min_silhouette,
        decomp_pca_dims_max=decomp_pca_dims_max,
        decomp_max_silhouette_samples=decomp_max_silhouette_samples,
        decomp_adaptive_k_min=decomp_adaptive_k_min, decomp_adaptive_k_max=decomp_adaptive_k_max,
        decomp_min_cluster_size_base=decomp_min_cluster_size_base,
        decomp_cluster_size_ratio=decomp_cluster_size_ratio,
        decomp_kmeans_n_init_min=decomp_kmeans_n_init_min,
        decomp_kmeans_n_init_max=decomp_kmeans_n_init_max,
        decomp_kmeans_scaling_factor=decomp_kmeans_scaling_factor)
    labels_np = np.array(labels)
    sizes = {int(k): int((labels_np == k).sum()) for k in range(n_concepts)}
    fragmented = n_concepts > _C.COMBO_OFFSET and sil >= min_silhouette
    return DecompositionTestResult(n_concepts, labels, sil, fragmented, sizes)

def test_editability(pos: torch.Tensor, neg: torch.Tensor, *, eot_temperature: float, default_score: float,
                     default_scale: float, eot_perturbation_scale: float,
                     eot_survival_weight: float, eot_spectral_weight: float) -> EditabilityTestResult:
    """Test EOT editability of the activation space."""
    from .editability import compute_eot_editability
    r = compute_eot_editability(pos, neg, temperature=eot_temperature, default_score=default_score,
        default_scale=default_scale, eot_perturbation_scale=eot_perturbation_scale,
        eot_survival_weight=eot_survival_weight, eot_spectral_weight=eot_spectral_weight)
    return EditabilityTestResult(
        composite_editability=r.composite_editability, steering_survival=r.steering_survival,
        spectral_concentration=r.spectral_concentration, spectral_sharpness=r.spectral_sharpness,
        attention_entropy=r.attention_entropy, jacobian_sensitivity=r.jacobian_sensitivity,
    )

def select_intervention(
    signal: SignalTestResult, geometry: GeometryTestResult,
    decomposition: DecompositionTestResult,
    editability: Optional[EditabilityTestResult] = None, *,
    zwiad_score_primary: float, zwiad_score_secondary: float, zwiad_score_tertiary: float,
    zwiad_editability_threshold: float, zwiad_przelom_bonus_max: float,
) -> InterventionResult:
    """Step four: Select optimal intervention method."""
    _method_names = ["CAA", "Ostrze", "MLP", "TECZA", "TETNO", "GROM", "Concept Flow", "PRZELOM"]
    _all = {m: zwiad_score_tertiary for m in _method_names}
    _nosig_msg = f"No signal (p > {STAT_ALPHA})"
    if not signal.passed:
        return InterventionResult("NONE", zwiad_score_tertiary, [_nosig_msg], _all)
    reasoning = [
        f"geometry: {geometry.diagnosis}",
        f"n_concepts: {decomposition.n_concepts}, fragmented: {decomposition.is_fragmented}",
        f"Z: {signal.max_z_score:.2f}, p: {signal.min_p_value:.4f}, gap: {geometry.gap:.3f}",
    ]
    if editability:
        reasoning.append(f"editability: {editability.composite_editability:.2f}")
    return InterventionResult("PENDING", zwiad_score_tertiary, reasoning, _all)

_GEO_FIELDS = ["linear_accuracy", "nonlinear_accuracy", "gap", "diagnosis", "confidence", "p_value",
    "gap_ci_lower", "gap_ci_upper", "n_diagnostics_passed", "n_diagnostics_total", "t_statistic",
    "residual_silhouette", "residuals_cluster", "ramsey_improvement", "ramsey_significant", "diagnostics"]

def run_full_protocol(
    pos: torch.Tensor, neg: torch.Tensor, device: str,
    model=None, tokenizer=None,
    layer: int = None, signal_keys: Optional[List[str]] = None,
    p_threshold: float = STAT_ALPHA, gap_threshold: Optional[float] = None,
    min_silhouette: Optional[float] = None, include_dimensionality_diagnostics: bool = True,
    shrinkage_low: float | None = None, shrinkage_moderate: float | None = None,
    shrinkage_high: float | None = None,
    power_excellent_threshold: float | None = None, power_low_threshold: float | None = None,
    mde_small_threshold: float | None = None, mde_medium_threshold: float | None = None,
    mde_large_threshold: float | None = None,
    *, decomp_pca_dims_max: int, decomp_max_silhouette_samples: int,
    decomp_adaptive_k_min: int, decomp_adaptive_k_max: int,
    decomp_min_cluster_size_base: int, decomp_cluster_size_ratio: int,
    decomp_kmeans_n_init_min: int, decomp_kmeans_n_init_max: int, decomp_kmeans_scaling_factor: int,
    zwiad_gap_k: float, zwiad_gap_min: float, zwiad_gap_max: float,
    default_scale: float, zwiad_sil_base: float, zwiad_sil_slope: float,
    zwiad_sil_scale_samples: float, zwiad_sil_max: float,
    zwiad_score_primary: float, zwiad_score_secondary: float, zwiad_score_tertiary: float,
    zwiad_editability_threshold: float, zwiad_przelom_bonus_max: float,
    default_score: float, blend_default: float,
    zwiad_ranges: Dict[str, float], zwiad_weights: Dict[str, float],
    eot_temperature: float, eot_perturbation_scale: float,
    eot_survival_weight: float, eot_spectral_weight: float,
    topology_max_dim: Optional[int] = None,
    topology_max_edge_length: Optional[float] = None,
    topology_persistence_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run complete Zwiad protocol."""
    if signal_keys is None:
        signal_keys = ["knn_accuracy", "knn_pca_accuracy", "mlp_probe_accuracy"]
    n_samples = len(pos)
    if gap_threshold is None:
        gap_threshold = adaptive_gap_threshold(
            n_samples, zwiad_gap_k=zwiad_gap_k, zwiad_gap_min=zwiad_gap_min, zwiad_gap_max=zwiad_gap_max)
    if min_silhouette is None:
        min_silhouette = adaptive_min_silhouette(
            n_samples, default_scale=default_scale, zwiad_sil_base=zwiad_sil_base,
            zwiad_sil_slope=zwiad_sil_slope, zwiad_sil_scale_samples=zwiad_sil_scale_samples, zwiad_sil_max=zwiad_sil_max)
    dim_diag = None
    if include_dimensionality_diagnostics:
        for _pname in ["shrinkage_low", "shrinkage_moderate", "shrinkage_high",
                       "power_excellent_threshold", "power_low_threshold",
                       "mde_small_threshold", "mde_medium_threshold", "mde_large_threshold"]:
            if locals()[_pname] is None:
                raise ValueError(f"{_pname} is required when include_dimensionality_diagnostics=True")
        from ..validation.dimensionality import run_dimensionality_diagnostics
        dim_diag = run_dimensionality_diagnostics(
            pos, neg, shrinkage_low=shrinkage_low, shrinkage_moderate=shrinkage_moderate, shrinkage_high=shrinkage_high,
            power_excellent_threshold=power_excellent_threshold, power_low_threshold=power_low_threshold,
            mde_small_threshold=mde_small_threshold, mde_medium_threshold=mde_medium_threshold,
            mde_large_threshold=mde_large_threshold,
        )
    _dk = {k: v for k, v in locals().items() if k.startswith("decomp_")}
    sig = test_signal(pos, neg, signal_keys, device, model, tokenizer, layer, p_threshold)
    geo = test_geometry(pos, neg, diagnostics_total_checks=DIAGNOSTICS_TOTAL_CHECKS)
    dec = test_decomposition(pos, neg, min_silhouette, **_dk)
    edit = test_editability(pos, neg, eot_temperature=eot_temperature, default_score=default_score,
        default_scale=default_scale, eot_perturbation_scale=eot_perturbation_scale,
        eot_survival_weight=eot_survival_weight, eot_spectral_weight=eot_spectral_weight)
    inter = select_intervention(sig, geo, dec, editability=edit,
        zwiad_score_primary=zwiad_score_primary, zwiad_score_secondary=zwiad_score_secondary,
        zwiad_score_tertiary=zwiad_score_tertiary, zwiad_editability_threshold=zwiad_editability_threshold,
        zwiad_przelom_bonus_max=zwiad_przelom_bonus_max)
    topo = None
    if topology_max_dim is not None and topology_max_edge_length is not None and topology_persistence_threshold is not None:
        topo = _test_topology(pos, neg, topology_max_dim, topology_max_edge_length, topology_persistence_threshold)
    result = {
        "protocol_config": {"n_samples": n_samples, "p_threshold": p_threshold,
            "gap_threshold": gap_threshold, "min_silhouette": min_silhouette},
        "signal_test": {"max_z_score": sig.max_z_score, "min_p_value": sig.min_p_value,
            "passed": sig.passed, "permutation_metrics": sig.permutation_metrics,
            "nonsense_metrics": sig.nonsense_metrics},
        "geometry_test": {f: getattr(geo, f) for f in _GEO_FIELDS},
        "decomposition_test": {"n_concepts": dec.n_concepts, "silhouette_score": dec.silhouette_score,
            "min_silhouette": min_silhouette, "is_fragmented": dec.is_fragmented,
            "per_concept_sizes": dec.per_concept_sizes, "cluster_labels": dec.cluster_labels},
        "intervention": {"recommended_method": inter.recommended_method, "confidence": inter.confidence,
            "reasoning": inter.reasoning, "method_scores": inter.method_scores},
        "editability_test": {"composite_editability": edit.composite_editability,
            "steering_survival": edit.steering_survival, "spectral_concentration": edit.spectral_concentration,
            "spectral_sharpness": edit.spectral_sharpness, "attention_entropy": edit.attention_entropy,
            "jacobian_sensitivity": edit.jacobian_sensitivity},
    }
    if topo is not None:
        result["topology_test"] = _topology_to_dict(topo)
        _b = topo.betti_union
        if (len(_b) > 1 and _b[1] > 0) or (len(_b) > 2 and _b[2] > 0):
            result["intervention"].update({"topology_hint": "manifold_topology_detected_prefer_nurt_szlak",
                "topology_betti": list(_b), "topology_named_shape": topo.named_shape_union})
    try:
        from .geometry_types import profile_benchmark
        geo_dict = {f: getattr(geo, f) for f in _GEO_FIELDS}
        geo_dict["n_concepts"] = dec.n_concepts
        result["geometry_profile"] = profile_benchmark(
            geo_dict, default_score=default_score, blend_default=blend_default,
            default_scale=default_scale, zwiad_ranges=zwiad_ranges, zwiad_weights=zwiad_weights).to_dict()
    except Exception:
        pass
    if dim_diag is not None:
        from dataclasses import asdict
        result["dimensionality_diagnostics"] = asdict(dim_diag)
    return result


run_zwiad_protocol = run_full_protocol
