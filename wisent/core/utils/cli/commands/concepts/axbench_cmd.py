"""Execute the `wisent axbench` command (steer | detect)."""

import json
import os
from typing import List

from wisent.core.utils.config_tools.constants import JSON_INDENT

__all__ = ["execute_axbench"]

_ACTION_STEER = "steer"
_ACTION_DETECT = "detect"


def _resolve_concept_ids(task, args) -> List[int]:
    if args.concept_id is not None:
        return [int(args.concept_id)]
    if args.all_concepts:
        concept_ids = task.list_concept_ids()
        if args.max_concepts:
            concept_ids = concept_ids[: args.max_concepts]
        return concept_ids
    raise ValueError("Pass --concept-id N or --all-concepts to select concepts.")


def _mean_of(records: List[dict], key: str) -> float:
    if not records:
        raise ValueError("No per-concept results were produced.")
    return sum(record[key] for record in records) / len(records)


def execute_axbench(args):
    """Run the AxBench protocol (steering or detection) over concepts."""
    from wisent.core.control.tasks.concepts.axbench_task import AxBenchTask
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.utils.cli.commands.concepts.axbench import (
        run_concept_detection,
        run_concept_steering,
    )

    action = args.axbench_action
    if action not in (_ACTION_STEER, _ACTION_DETECT):
        raise ValueError(f"Unknown --axbench-action '{action}'.")

    print(f"\n{'=' * 80}")
    print(f"🧭 AXBENCH {action.upper()} — variant={args.variant}, model={args.model}")
    print(f"{'=' * 80}\n")

    task = AxBenchTask(variant=args.variant, subset=args.concept_set)
    concept_ids = _resolve_concept_ids(task, args)
    print(f"   Concepts: {concept_ids}\n")

    work_dir = args.work_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output)) or ".", "axbench_work",
    )
    os.makedirs(work_dir, exist_ok=True)

    judge = None
    if action == _ACTION_STEER:
        from wisent.core.utils.services.judges import make_judge

        judge = make_judge(
            args.judge_model,
            device=args.device,
            batch_size=args.judge_batch_size,
            max_new_tokens=args.judge_max_new_tokens,
            temperature=args.judge_temperature,
        )

    print(f"📦 Loading subject model '{args.model}'...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Loaded ({model.num_layers} layers)\n")

    records = []
    for concept_id in concept_ids:
        print(f"▶ Concept {concept_id}", flush=True)
        if action == _ACTION_STEER:
            record = run_concept_steering(
                task, concept_id, model, args.model, args, judge, work_dir,
            )
            print(
                f"   ✓ concept {concept_id}: overall={record['overall']:.3f} "
                f"(c={record['concept_score']:.2f}, i={record['instruct_score']:.2f}, "
                f"f={record['fluency_score']:.2f}, factor={record['best_factor']})\n",
                flush=True,
            )
        else:
            record = run_concept_detection(
                task, concept_id, model, args.model, args, work_dir,
            )
            print(
                f"   ✓ concept {concept_id}: auroc={record['auroc']:.3f} "
                f"f1={record['f1']:.3f}\n",
                flush=True,
            )
        records.append(record)

    summary = {
        "task": task.get_name(),
        "model": args.model,
        "action": action,
        "method": args.method,
        "num_concepts": len(records),
        "per_concept": records,
    }
    if action == _ACTION_STEER:
        summary["judge_model"] = args.judge_model
        summary["mean_overall"] = _mean_of(records, "overall")
        summary["mean_concept_score"] = _mean_of(records, "concept_score")
        summary["mean_instruct_score"] = _mean_of(records, "instruct_score")
        summary["mean_fluency_score"] = _mean_of(records, "fluency_score")
    else:
        summary["mean_auroc"] = _mean_of(records, "auroc")
        summary["mean_f1"] = _mean_of(records, "f1")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=JSON_INDENT)

    print(f"{'=' * 80}")
    print(f"✅ AXBENCH {action.upper()} COMPLETE — results: {args.output}")
    if action == _ACTION_STEER:
        print(f"   Mean overall (harmonic): {summary['mean_overall']:.3f}")
    else:
        print(f"   Mean AUROC: {summary['mean_auroc']:.3f}")
    print(f"{'=' * 80}\n")
