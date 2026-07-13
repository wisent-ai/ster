"""Intervention Validation for Zwiad.

Tests whether Zwiad diagnosis predicts CAA steering success.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import random

from wisent.core import constants as _C
from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    RECURSION_INITIAL_DEPTH,
    HASH_PREFIX_LEN,
    ARCHITECTURE_MODULE_LIMIT_DEFAULT,
)
from wisent.examples.scripts.intervention_validation_helpers import (
    gcs_upload_file,
    SteeringResult,
    ValidationResults,
    compute_caa_direction,
    load_diagnosis_results,
    get_diagnosis_for_benchmark,
    convert_to_serializable,
)
from wisent.examples.scripts.intervention_validation_eval import (
    evaluate_steering,
    evaluate_baseline,
)


def run_intervention_validation(
    model_name: str,
    report_interval: int,
    benchmarks_to_test: Optional[List[str]] = None,
    *,
    samples_per_benchmark_count: int,
    test_samples_count: int,
    signal_exist_threshold: float,
    signal_probe_accuracy_threshold: float,
    signal_linear_gap: float,
    steering_coefficients=(1.0, 2.0, 5.0, 10.0),
):
    """
    Run intervention validation experiments.

    Args:
        model_name: Model to test
        benchmarks_to_test: Specific benchmarks (default: sample from each diagnosis)
        steering_coefficients: Coefficients to test
    """
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
    from wisent.core.primitives.model_interface.core.activations.activation_cache import ActivationCache, collect_and_cache_activations
    from lm_eval.tasks import TaskManager
    from wisent.extractors.lm_eval.lm_task_pairs_generation import lm_build_contrastive_pairs
    samples_per_benchmark = samples_per_benchmark_count
    test_samples = test_samples_count
    print("=" * _C.SEPARATOR_WIDTH_WIDE)
    print("INTERVENTION VALIDATION")
    print("=" * _C.SEPARATOR_WIDTH_WIDE)
    print(f"Model: {model_name}")

    output_dir = Path("/tmp/intervention_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_results = load_diagnosis_results(model_name, output_dir)
    if not diagnosis_results:
        print("ERROR: No diagnosis results found. Run discover_directions first.")
        return

    if benchmarks_to_test is None:
        benchmarks_to_test = []
        by_diagnosis = {"LINEAR": [], "NONLINEAR": [], "NO_SIGNAL": []}

        for category, data in diagnosis_results.items():
            results = data.get("results", [])
            seen_benchmarks = set()
            for r in results:
                bench = r["benchmark"]
                if bench in seen_benchmarks:
                    continue
                seen_benchmarks.add(bench)
                signal = r["signal_strength"]
                linear = r["linear_probe_accuracy"]
                if signal < signal_exist_threshold:
                    by_diagnosis["NO_SIGNAL"].append(bench)
                elif linear > signal_probe_accuracy_threshold and (signal - linear) < signal_linear_gap:
                    by_diagnosis["LINEAR"].append(bench)
                else:
                    by_diagnosis["NONLINEAR"].append(bench)

        random.seed(_C.DEFAULT_RANDOM_SEED)
        for diag, benches in by_diagnosis.items():
            if benches:
                sampled = random.sample(benches, min(3, len(benches)))
                benchmarks_to_test.extend(sampled)
                print(f"  {diag}: {sampled}")

    print(f"\nBenchmarks to test: {benchmarks_to_test}")

    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name, device="cuda")
    print(f"  Layers: {model.num_layers}, Hidden: {model.hidden_size}")

    model_prefix = model_name.replace('/', '_')
    cache_dir = f"/tmp/wisent_intervention_cache_{model_prefix}"
    cache = ActivationCache(cache_dir, hash_digest_prefix=HASH_PREFIX_LEN)

    validation_results = ValidationResults(model=model_name)

    tm = TaskManager()
    strategy = ExtractionStrategy.default()

    for benchmark in benchmarks_to_test:
        print(f"\n{'-' * _C.SEPARATOR_WIDTH_MEDIUM}")
        print(f"Benchmark: {benchmark}")
        print("-" * _C.SEPARATOR_WIDTH_MEDIUM)

        diagnosis, best_layer, signal, linear_acc = get_diagnosis_for_benchmark(
            diagnosis_results, benchmark, strategy.value
        )
        print(f"  Diagnosis: {diagnosis}")
        print(f"  Signal: {signal:.3f}, Linear: {linear_acc:.3f}")
        print(f"  Best layer: {best_layer}")

        try:
            task_dict = tm.load_task_or_group([benchmark])
            task = list(task_dict.values())[RECURSION_INITIAL_DEPTH]
        except Exception:
            task = None

        try:
            all_pairs = lm_build_contrastive_pairs(
                benchmark,
                task,
                limit=samples_per_benchmark + test_samples,
                train_ratio=args.train_ratio,
            )
        except Exception as e:
            print(f"  ERROR loading pairs: {e}")
            continue

        if len(all_pairs) < samples_per_benchmark + test_samples:
            print(f"  SKIP: Not enough pairs ({len(all_pairs)})")
            continue

        random.shuffle(all_pairs)
        train_pairs = all_pairs[:samples_per_benchmark]
        test_pairs = all_pairs[samples_per_benchmark:samples_per_benchmark + test_samples]

        print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
        print(f"  Extracting activations...")

        try:
            cached = collect_and_cache_activations(
                model=model,
                pairs=train_pairs,
                benchmark=benchmark,
                strategy=strategy,
                report_interval=report_interval,
                cache=cache,
                show_progress=False,
                architecture_module_limit=ARCHITECTURE_MODULE_LIMIT_DEFAULT,
            )
        except Exception as e:
            print(f"  ERROR extracting activations: {e}")
            continue

        layer_name = str(best_layer + COMBO_OFFSET)
        try:
            pos_acts = cached.get_positive_activations(layer_name)
            neg_acts = cached.get_negative_activations(layer_name)
        except Exception as e:
            print(f"  ERROR getting activations: {e}")
            continue

        direction = compute_caa_direction(pos_acts, neg_acts)
        print(f"  Direction norm: {direction.norm().item():.4f}")

        print(f"  Evaluating baseline...")
        base_acc, base_correct_lp, base_incorrect_lp = evaluate_baseline(model, test_pairs)
        print(f"    Baseline accuracy: {base_acc:.3f}")
        print(f"    Baseline logprob gap: {base_correct_lp - base_incorrect_lp:.4f}")

        best_result = None
        best_improvement = -float('inf')

        for coef in steering_coefficients:
            print(f"  Testing coefficient={coef}...")
            steered_acc, steered_correct_lp, steered_incorrect_lp = evaluate_steering(
                model, test_pairs, best_layer, direction, coef
            )
            acc_change = steered_acc - base_acc
            lp_shift = (steered_correct_lp - steered_incorrect_lp) - (base_correct_lp - base_incorrect_lp)
            print(f"    Steered accuracy: {steered_acc:.3f} (change: {acc_change:+.3f})")
            print(f"    Logprob shift: {lp_shift:+.4f}")
            steering_success = (
                acc_change > 0.05
                or lp_shift > 0.1
            )
            if acc_change > best_improvement:
                best_improvement = acc_change
                best_result = SteeringResult(
                    benchmark=benchmark,
                    strategy=strategy.value,
                    layer=best_layer,
                    diagnosis=diagnosis,
                    baseline_accuracy=base_acc,
                    baseline_correct_logprob=base_correct_lp,
                    baseline_incorrect_logprob=base_incorrect_lp,
                    steered_accuracy=steered_acc,
                    steered_correct_logprob=steered_correct_lp,
                    steered_incorrect_logprob=steered_incorrect_lp,
                    accuracy_change=acc_change,
                    logprob_shift=lp_shift,
                    steering_success=steering_success,
                    steering_coefficient=coef,
                    num_test_samples=len(test_pairs),
                )

        if best_result:
            validation_results.results.append(best_result)
            print(f"\n  Best result: coef={best_result.steering_coefficient}, "
                  f"acc_change={best_result.accuracy_change:+.3f}, "
                  f"success={best_result.steering_success}")

    validation_results.compute_summary()

    print("\n" + "=" * _C.SEPARATOR_WIDTH_WIDE)
    print("VALIDATION SUMMARY")
    print("=" * _C.SEPARATOR_WIDTH_WIDE)
    print(f"\nLinear diagnosis -> CAA success rate: {validation_results.linear_success_rate:.1%}")
    print(f"Nonlinear diagnosis -> CAA success rate: {validation_results.nonlinear_success_rate:.1%}")
    print(f"No signal diagnosis -> CAA success rate: {validation_results.no_signal_success_rate:.1%}")

    if validation_results.linear_success_rate > validation_results.nonlinear_success_rate:
        print("\n✓ VALIDATION PASSED: LINEAR diagnosis predicts higher CAA success!")
    else:
        print("\n✗ VALIDATION FAILED: LINEAR diagnosis does not predict higher CAA success")

    results_file = output_dir / f"{model_prefix}_validation.json"
    results_dicts = [convert_to_serializable(asdict(r)) for r in validation_results.results]

    with open(results_file, "w") as f:
        json.dump({
            "model": model_name,
            "results": results_dicts,
            "summary": {
                "linear_success_rate": float(validation_results.linear_success_rate),
                "nonlinear_success_rate": float(validation_results.nonlinear_success_rate),
                "no_signal_success_rate": float(validation_results.no_signal_success_rate),
            }
        }, f, indent=_C.JSON_INDENT)

    print(f"\nResults saved to: {results_file}")
    gcs_upload_file(results_file, model_name)

    del model

    return validation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervention validation for Zwiad")
    parser.add_argument("--model", type=str, required=True, help="Model to test")
    parser.add_argument("--report-interval", type=int, required=True, help="Progress reporting interval for activation collection")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=None, help="Benchmarks to test")
    parser.add_argument("--samples-per-benchmark", type=int, required=True, help="Samples per benchmark")
    parser.add_argument("--test-samples", type=int, required=True, help="Test samples count")
    parser.add_argument("--signal-exist-threshold", type=float, required=True, help="Signal existence threshold")
    parser.add_argument("--signal-probe-accuracy-threshold", type=float, required=True, help="Signal probe accuracy threshold")
    parser.add_argument("--signal-linear-gap", type=float, required=True, help="Signal linear gap threshold")
    args = parser.parse_args()

    run_intervention_validation(
        model_name=args.model,
        report_interval=args.report_interval,
        benchmarks_to_test=args.benchmarks,
        samples_per_benchmark_count=args.samples_per_benchmark,
        test_samples_count=args.test_samples,
        signal_exist_threshold=args.signal_exist_threshold,
        signal_probe_accuracy_threshold=args.signal_probe_accuracy_threshold,
        signal_linear_gap=args.signal_linear_gap,
    )
