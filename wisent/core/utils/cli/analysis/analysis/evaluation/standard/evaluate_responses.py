"""Evaluate responses command execution logic."""

import json
import os
from pathlib import Path

from wisent.core.utils.config_tools.constants import JSON_INDENT
from wisent.core.reading.evaluators.core.rotator import EvaluatorRotator

from wisent.core.utils.cli.analysis.analysis.evaluation.specialized.evaluate_axbench import evaluate_axbench
from wisent.core.utils.cli.analysis.analysis.evaluation.specialized.evaluate_docker import evaluate_docker_execution
from wisent.core.utils.cli.analysis.analysis.evaluation.specialized.evaluate_personalization import evaluate_personalization
from wisent.core.utils.cli.analysis.analysis.evaluation.standard.evaluate_refusal import evaluate_refusal
from wisent.core.utils.cli.analysis.analysis.evaluation.standard.evaluate_standard import evaluate_standard


def _load_task_evaluator_json() -> dict:
    """Find and load task-evaluator.json, return tasks dict or empty dict."""
    current_dir = Path(__file__).resolve()
    for parent in current_dir.parents:
        task_eval_file = parent / 'task-evaluator.json'
        if task_eval_file.exists():
            with open(task_eval_file) as f:
                return json.load(f).get('tasks', {})

    import wisent
    pkg_task_eval = Path(wisent.__file__).resolve().parent / "task-evaluator.json"
    if pkg_task_eval.exists():
        with open(pkg_task_eval) as f:
            return json.load(f).get('tasks', {})

    return {}


def _get_special_evaluation_type(task_name: str) -> str | None:
    """Check if task requires special evaluation (docker, personalization, refusal).

    Returns evaluation_type string if special, None for standard evaluation.
    Tries exact match first, then longest-prefix match against task-evaluator.json
    keys (so e.g. "humaneval_instruct" matches "humaneval").
    """
    tasks = _load_task_evaluator_json()
    task_config = tasks.get(task_name)
    if not task_config:
        # Longest-prefix match: pick the longest key that is a prefix of task_name.
        best_key = None
        for key in tasks:
            if task_name.startswith(key) and (best_key is None or len(key) > len(best_key)):
                best_key = key
        if best_key is None:
            return None
        task_config = tasks[best_key]
    evaluation_type = task_config.get('evaluation_type')
    if evaluation_type in ("docker_execution", "personalization", "refusal", "axbench_judge"):
        return evaluation_type
    return None


def _load_task_docs(task_name: str, train_ratio: float):
    """Load ground truth docs for evaluation. Handles both lm-eval and HuggingFace benchmarks."""
    from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor
    from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

    extractor = get_extractor(task_name)

    if isinstance(extractor, HuggingFaceBenchmarkExtractor):
        pairs = extractor.extract_contrastive_pairs(limit=None)
        task_docs = []
        for pair in pairs:
            task_docs.append({
                "question": pair.prompt,
                "answer": pair.positive_response.model_response,
                "positive_reference": pair.positive_response.model_response,
                "negative_reference": pair.negative_response.model_response,
            })
        print(f"   ✓ Loaded {len(task_docs)} docs from HuggingFace\n")
        return task_docs
    else:
        from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
        from wisent.core.utils import get_all_docs_from_task, create_deterministic_split
        from wisent.extractors.lm_eval.registry.lm_task_pairs_generation import _flatten_task_dict
        from lm_eval.api.task import ConfigurableTask

        task_obj = LMEvalDataLoader.load_lm_eval_task(task_name)

        if isinstance(task_obj, ConfigurableTask):
            task = task_obj
        else:
            # Group task dict — pick first leaf ConfigurableTask
            leaf_tasks = _flatten_task_dict(task_obj)
            _, task = leaf_tasks[0]

        all_docs, split_counts = get_all_docs_from_task(task)
        _, task_docs = create_deterministic_split(all_docs, task_name, train_ratio=train_ratio)

        print(f"   ✓ Combined {len(all_docs)} total docs from splits: {split_counts}")
        print(f"   ✓ Using TEST portion: {len(task_docs)} task documents for evaluation\n")
        return task_docs


def execute_evaluate_responses(args):
    """Execute the evaluate-responses command.

    Evaluates generated responses using benchmark-specific evaluators.
    Uses EvaluatorRotator to auto-select the appropriate evaluator based on task name.
    """
    import wisent.core.reading.evaluators.core.benchmark_specific  # noqa: F401

    model_name = getattr(args, 'model', None)
    if model_name and not getattr(args, 'cached_model', None):
        from wisent.core.primitives.models.core.wisent_model import WisentModel
        print(f"Loading model '{model_name}'...")
        args.cached_model = WisentModel(model_name, device=getattr(args, 'device', None))
        print(f"Model loaded.\n")

    print(f"\n{'='*80}")
    print(f"📊 EVALUATING GENERATED RESPONSES")
    print(f"{'='*80}")
    print(f"   Input: {args.input}")
    print(f"{'='*80}\n")

    # Load input file
    print(f"📂 Loading generated responses...")
    with open(args.input, 'r') as f:
        input_data = json.load(f)

    if isinstance(input_data, list):
        responses = input_data
        task_name = args.task if args.task else "generic"
    else:
        responses = input_data.get('responses', [])
        task_name = args.task if args.task else input_data.get('task', "generic")

    if not task_name:
        raise ValueError("Task name not found in input file and not provided via --task")
    print(f"   ✓ Loaded {len(responses)} responses")
    print(f"   Task: {task_name}\n")

    # Check for special evaluation types (docker, personalization, refusal)
    special_type = _get_special_evaluation_type(task_name)

    if special_type == "docker_execution":
        evaluation_results = []
        task_results = []
        task_config = _load_task_evaluator_json().get(task_name, {})
        aggregated_metrics = evaluate_docker_execution(
            args, input_data, responses, task_name, evaluation_results, task_results,
            subprocess_timeout=args.subprocess_timeout,
            task_config=task_config)
        if aggregated_metrics is not None:
            return

    if special_type == "personalization":
        evaluation_results = []
        task_results = []
        aggregated_metrics = evaluate_personalization(
            args, input_data, responses, task_name, evaluation_results, task_results,
            personalization_good_threshold=args.personalization_good_threshold)
        if aggregated_metrics is not None:
            return

    if special_type == "refusal":
        evaluation_results = []
        task_results = []
        aggregated_metrics = evaluate_refusal(
            args, input_data, responses, task_name, evaluation_results, task_results)
        if aggregated_metrics is not None:
            return

    if special_type == "axbench_judge":
        evaluation_results = []
        task_results = []
        aggregated_metrics = evaluate_axbench(
            args, input_data, responses, task_name, evaluation_results, task_results)
        if aggregated_metrics is not None:
            return

    # Standard evaluation — use EvaluatorRotator
    # task_docs are only needed when responses lack 'positive_reference'. Skip loading
    # them when every response already carries positive_reference (which is the case
    # for pairs piped through generate-pairs-from-task → evaluate-responses).
    needs_task_docs = any(r.get("positive_reference") is None for r in responses)
    if needs_task_docs:
        print(f"📚 Loading task data...")
        task_docs = _load_task_docs(task_name, train_ratio=getattr(args, 'train_ratio', 0.5))
    else:
        task_docs = []

    print(f"🔧 Selecting evaluator for task '{task_name}'...")
    rotator = EvaluatorRotator(
        task_name=task_name,
        evaluator_kwargs={
            "f1_threshold": args.f1_threshold,
            "generation_embedding_weight": args.generation_embedding_weight,
            "generation_nli_weight": args.generation_nli_weight,
        },
    )
    evaluator = rotator.current
    print(f"   Using: {evaluator.name} (auto-selected via EvaluatorRotator)\n")

    # Derive evaluation_type for evaluate_standard routing
    from wisent.core.utils.cli.analysis.analysis.evaluation.standard.eval_config_resolver import derive_eval_config_from_extractor
    evaluation_type, _ = derive_eval_config_from_extractor(task_name)

    # Evaluate
    print(f"🎯 Evaluating responses...\n")
    evaluation_results = []
    task_results = []

    evaluate_standard(
        args, input_data, responses, task_name,
        evaluation_results, task_results, evaluator, task_docs,
        evaluation_type=evaluation_type,
        model=getattr(args, 'cached_model', None),
    )

    # Aggregate results
    aggregated_metrics = {}
    if task_results:
        all_metric_keys = set()
        for result in task_results:
            all_metric_keys.update(result.keys())
        for metric_key in all_metric_keys:
            values = [r[metric_key] for r in task_results if metric_key in r]
            if values:
                aggregated_metrics[metric_key] = sum(values) / len(values)

    # Save results
    print(f"\n💾 Saving evaluation results...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Count model_required skips
    num_model_required = sum(
        1 for er in evaluation_results
        if er.get("evaluation", {}).get("error") == "model_required"
    )

    output_data = {
        "input_file": args.input,
        "task": task_name if isinstance(input_data, list) else input_data.get('task'),
        "model": None if isinstance(input_data, list) else input_data.get('model'),
        "evaluator_used": evaluator.name,
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "num_model_required": num_model_required,
        "evaluations": evaluation_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)

    print(f"   ✓ Results saved to: {args.output}\n")

    print(f"{'='*80}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Evaluator: {evaluator.name}")
    print(f"   Total responses: {len(responses)}")
    print(f"   Successfully evaluated: {len(task_results)}")
    print(f"   Failed/skipped: {len(responses) - len(task_results)}")
    print(f"\n   Metrics:")
    for metric_name, metric_value in aggregated_metrics.items():
        if isinstance(metric_value, float):
            print(f"     {metric_name}: {metric_value:.4f}")
        else:
            print(f"     {metric_name}: {metric_value}")
    print(f"{'='*80}\n")
