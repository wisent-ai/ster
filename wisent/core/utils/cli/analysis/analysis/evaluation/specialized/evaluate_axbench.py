"""AxBench judge evaluation for the evaluate-responses command.

Scores generated responses with the verbatim AxBench rubrics (concept
incorporation, instruction relatedness, fluency, each 0-2) and aggregates
with the harmonic mean + hard-zero rule. Routed via the 'axbench_judge'
evaluation_type in task-evaluator.json.
"""

import json
import os
from collections import defaultdict

from wisent.core.utils.config_tools.constants import JSON_INDENT

__all__ = ["evaluate_axbench"]


def _resolve_concept(args, input_data, response) -> str:
    """Concept label precedence: --concept-label, response metadata, file field."""
    explicit = getattr(args, "concept_label", None)
    if explicit:
        return explicit
    metadata = response.get("metadata") or {}
    if metadata.get("concept"):
        return str(metadata["concept"])
    if isinstance(input_data, dict) and input_data.get("concept"):
        return str(input_data["concept"])
    raise ValueError(
        "Cannot resolve the AxBench concept label: pass --concept-label, or "
        "provide 'concept' in each response's metadata or at the top level "
        "of the input file."
    )


def evaluate_axbench(args, input_data, responses, task_name, evaluation_results, task_results):
    """Judge-score responses per the AxBench protocol. Returns aggregated metrics."""
    from wisent.core.utils.services.judges import make_judge
    from wisent.core.utils.cli.commands.concepts.axbench import score_generations

    judge_model = getattr(args, "judge_model", None)
    judge = make_judge(
        judge_model,
        device=getattr(args, "device", None),
        batch_size=args.judge_batch_size,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.judge_temperature,
    )
    print(f"⚖️  AxBench judge evaluation using '{judge_model}' on {len(responses)} responses\n")

    valid = [r for r in responses if r.get("generated_response")]
    if not valid:
        raise ValueError("No responses with a 'generated_response' field to judge.")

    groups = defaultdict(list)
    for response in valid:
        groups[_resolve_concept(args, input_data, response)].append(response)

    for concept_label, group in groups.items():
        generations = [str(r["generated_response"]) for r in group]
        instructions = [str(r.get("prompt", "")) for r in group]
        if not all(instructions):
            raise ValueError(
                f"Responses for concept '{concept_label}' are missing 'prompt' "
                "fields; the instruct rubric requires the original instruction."
            )
        scores = score_generations(generations, instructions, concept_label, judge)
        for i, response in enumerate(group):
            record = {
                "concept": concept_label,
                "prompt": instructions[i],
                "generated_response": generations[i],
                "concept_score": scores["concept_scores"][i],
                "instruct_score": scores["instruct_scores"][i],
                "fluency_score": scores["fluency_scores"][i],
                "overall": scores["overall_scores"][i],
            }
            evaluation_results.append(record)
            task_results.append({
                "concept_score": record["concept_score"],
                "instruct_score": record["instruct_score"],
                "fluency_score": record["fluency_score"],
                "overall": record["overall"],
            })

    aggregated_metrics = {
        key: sum(r[key] for r in task_results) / len(task_results)
        for key in ("concept_score", "instruct_score", "fluency_score", "overall")
    }

    print(f"💾 Saving evaluation results...")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_data = {
        "input_file": args.input,
        "task": task_name if isinstance(input_data, list) else input_data.get("task"),
        "model": None if isinstance(input_data, list) else input_data.get("model"),
        "evaluation_type": "axbench_judge",
        "judge_model": judge_model,
        "aggregated_metrics": aggregated_metrics,
        "num_evaluated": len(task_results),
        "num_total": len(responses),
        "evaluations": evaluation_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=JSON_INDENT)
    print(f"   ✓ Results saved to: {args.output}\n")

    print(f"{'=' * 80}")
    print(f"✅ AXBENCH JUDGE EVALUATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"   Concepts: {len(groups)}, responses judged: {len(task_results)}")
    print(f"   Mean overall (harmonic): {aggregated_metrics['overall']:.4f}")
    print(f"   Mean concept/instruct/fluency: "
          f"{aggregated_metrics['concept_score']:.3f} / "
          f"{aggregated_metrics['instruct_score']:.3f} / "
          f"{aggregated_metrics['fluency_score']:.3f}")
    print(f"{'=' * 80}\n")
    return aggregated_metrics
