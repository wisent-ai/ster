"""
Pairs ablation and optimal configuration finding for direction discovery.
"""

from typing import Dict, List, Optional

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.examples.scripts._discovery_utils import OptimalConfig


def run_pairs_ablation(
    runner: "GeometryRunner",
    benchmark: str,
    layer: int,
    strategy: "ExtractionStrategy",
    *,
    pair_counts: List[int],
) -> Dict[int, float]:
    """
    Run ablation study on number of contrastive pairs.
    
    Tests different numbers of pairs to find the minimum needed for stable signal.
    
    Args:
        runner: GeometryRunner with loaded model
        benchmark: Benchmark name
        layer: Layer to test
        strategy: Extraction strategy
        pair_counts: List of pair counts to test (default: [10, 25, 50, 100, 200])
        
    Returns:
        Dict mapping n_pairs -> accuracy
    """
    from wisent.core.reading.modules.runner.geometry_runner import compute_geometry_metrics
    from wisent.core.primitives.model_interface.core.activations.activation_cache import CachedActivations
    
    pair_counts = list(pair_counts)
    results = {}
    
    # Get full cached activations
    try:
        cached = runner._get_cached_activations(benchmark, strategy, show_progress=False)
    except Exception:
        return {}
    
    layer_name = str(layer)
    if layer_name not in cached.get_available_layers():
        return {}
    
    max_pairs = cached.num_pairs
    
    for n in pair_counts:
        if n > max_pairs:
            continue
        
        # Create subsampled CachedActivations
        sub_cached = CachedActivations(
            benchmark=cached.benchmark,
            strategy=cached.strategy,
            model_name=cached.model_name,
            num_layers=cached.num_layers,
            model_revision=cached.model_revision,
            tokenizer_revision=cached.tokenizer_revision,
            hidden_size=cached.hidden_size,
        )
        sub_cached.pair_activations = cached.pair_activations[:n]
        sub_cached.num_pairs = n
        
        # Compute metrics
        try:
            result = compute_geometry_metrics(sub_cached, (layer,))
            results[n] = result.linear_probe_accuracy
        except Exception:
            pass
    
    return results


def find_optimal_config(
    results: "GeometrySearchResults",
    *,
    pairs_ablation_default_min: int,
    nonsense_analysis: Optional[Dict[str, Dict]] = None,
    pairs_ablation: Optional[Dict[str, Dict[int, float]]] = None,
) -> Optional[OptimalConfig]:
    """
    Find optimal configuration from geometry search results.
    
    Args:
        results: GeometrySearchResults from runner
        nonsense_analysis: Per-benchmark nonsense baseline results
        pairs_ablation: Per-benchmark pairs ablation results
        
    Returns:
        OptimalConfig with optimal settings, or None if no signal
    """
    if not results.results:
        return None
    
    # Find best result by linear_probe_accuracy
    best_result = max(results.results, key=lambda r: r.linear_probe_accuracy)
    
    # Find optimal layer (single layer with best accuracy)
    layer_accuracies: Dict[int, List[float]] = {}
    for r in results.results:
        if len(r.layers) == 1:
            layer = r.layers[0]
            if layer not in layer_accuracies:
                layer_accuracies[layer] = []
            layer_accuracies[layer].append(r.linear_probe_accuracy)
    
    if layer_accuracies:
        layer_avg = {l: sum(accs)/len(accs) for l, accs in layer_accuracies.items()}
        optimal_layer = max(layer_avg, key=layer_avg.get)
        layer_accuracy = layer_avg[optimal_layer]
        
        # Find layers within 5% of best
        threshold = layer_accuracy * 0.95
        optimal_layer_range = sorted([l for l, acc in layer_avg.items() if acc >= threshold])[:3]
    else:
        optimal_layer = best_result.layers[0] if best_result.layers else 0
        layer_accuracy = best_result.linear_probe_accuracy
        optimal_layer_range = [optimal_layer]
    
    # Find optimal strategy
    strategy_accuracies: Dict[str, List[float]] = {}
    for r in results.results:
        if r.strategy not in strategy_accuracies:
            strategy_accuracies[r.strategy] = []
        strategy_accuracies[r.strategy].append(r.linear_probe_accuracy)
    
    if strategy_accuracies:
        strategy_avg = {s: sum(accs)/len(accs) for s, accs in strategy_accuracies.items()}
        optimal_strategy = max(strategy_avg, key=strategy_avg.get)
        strategy_accuracy = strategy_avg[optimal_strategy]
    else:
        optimal_strategy = best_result.strategy
        strategy_accuracy = best_result.linear_probe_accuracy
    
    # Aggregate pairs ablation
    pairs_saturation_curve: Dict[int, float] = {}
    if pairs_ablation:
        # Average across benchmarks
        all_n_pairs = set()
        for ablation in pairs_ablation.values():
            all_n_pairs.update(ablation.keys())
        
        for n in sorted(all_n_pairs):
            accs = [ablation[n] for ablation in pairs_ablation.values() if n in ablation]
            if accs:
                pairs_saturation_curve[n] = sum(accs) / len(accs)
    
    # Find minimum pairs for stable signal (within 5% of max)
    if pairs_saturation_curve:
        max_acc = max(pairs_saturation_curve.values())
        threshold = max_acc * 0.95
        min_pairs = min([n for n, acc in pairs_saturation_curve.items() if acc >= threshold], default=pairs_ablation_default_min)
    else:
        min_pairs = pairs_ablation_default_min  # default
    
    # Multi-direction analysis
    num_directions = int(best_result.multi_dir_min_k_for_good) if best_result.multi_dir_min_k_for_good > 0 else 1
    single_dir_acc = best_result.multi_dir_accuracy_k1
    multi_dir_acc = best_result.multi_dir_accuracy_k5
    steering_gain = multi_dir_acc - single_dir_acc
    
    # ICD and nonsense baseline
    if nonsense_analysis:
        valid_analyses = [a for a in nonsense_analysis.values() if a]
        if valid_analyses:
            avg_icd = sum(a["real_icd"]["icd"] for a in valid_analyses) / len(valid_analyses)
            avg_signal_above = sum(a["baseline_comparison"]["signal_above_baseline"] for a in valid_analyses) / len(valid_analyses)
            # Majority verdict
            from collections import Counter
            verdicts = [a["verdict"] for a in valid_analyses]
            verdict = Counter(verdicts).most_common(1)[0][0]
        else:
            avg_icd = 0.0
            avg_signal_above = 0.0
            verdict = "NO_DATA"
    else:
        avg_icd = 0.0
        avg_signal_above = 0.0
        verdict = "NOT_COMPUTED"
    
    # Determine if linear
    avg_linear = sum(r.linear_probe_accuracy for r in results.results) / len(results.results)
    avg_best_nonlinear = sum(
        max(r.knn_accuracy_k10, r.knn_pca_accuracy, r.knn_umap_accuracy, r.mlp_probe_accuracy) 
        for r in results.results
    ) / len(results.results)
    is_linear = avg_linear >= avg_best_nonlinear - 0.15
    
    return OptimalConfig(
        optimal_layer=optimal_layer,
        optimal_layer_range=optimal_layer_range,
        layer_accuracy=layer_accuracy,
        optimal_strategy=optimal_strategy,
        strategy_accuracy=strategy_accuracy,
        min_pairs_for_stable_signal=min_pairs,
        pairs_saturation_curve=pairs_saturation_curve,
        num_directions_needed=num_directions,
        single_direction_accuracy=single_dir_acc,
        multi_direction_accuracy=multi_dir_acc,
        steering_gain=steering_gain,
        icd=avg_icd,
        signal_above_noise=avg_signal_above,
        is_linear=is_linear,
        verdict=verdict,
    )

