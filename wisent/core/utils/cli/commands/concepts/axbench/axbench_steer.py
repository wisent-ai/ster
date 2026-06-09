"""AxBench model-steering protocol for one concept.

Protocol (arXiv 2501.17148): train a per-concept steering object on
Concept500 pairs, generate up to 128 tokens on Alpaca-Eval instructions over
the steering-factor grid, select the best factor on the select half by mean
judge harmonic score, then report judge scores on the held-out half.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

from wisent.core.control.steering_methods.configs.optimal import (
    get_optimal,
    get_optimal_extraction_strategy,
)
from wisent.core.control.tasks.concepts.axbench_pairs import (
    build_concept_pairs,
    pairs_to_json_doc,
)
from wisent.core.utils.cli.commands.concepts.axbench.alpaca_eval import (
    load_alpaca_instructions,
    split_select_eval,
)
from wisent.core.utils.cli.commands.concepts.axbench.axbench_judge import score_generations
from wisent.core.utils.config_tools.constants import JSON_INDENT

__all__ = ["train_concept_steering_object", "run_concept_steering"]


def _make_args(**kwargs) -> argparse.Namespace:
    """Create an argparse.Namespace from kwargs (pipeline.py pattern)."""
    args = argparse.Namespace()
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def train_concept_steering_object(
    task,
    concept_id: int,
    model,
    model_name: str,
    layer: int,
    args,
    work_dir: str,
) -> Tuple[str, str]:
    """Build pairs, collect activations and train the per-concept steering
    object. Returns (steering_object_path, concept_label)."""
    import wisent.core.utils.cli as cli

    concept = task.concept_label(concept_id)
    pos_rows = task.positive_rows(concept_id)
    neg_rows = task.negative_rows()
    hard_rows = task.hard_negative_rows(concept_id) if args.use_hard_negatives else None

    pairs = build_concept_pairs(
        pos_rows, neg_rows, concept_id, concept,
        hard_neg_rows=hard_rows, limit=args.pair_limit,
    )
    os.makedirs(work_dir, exist_ok=True)
    pairs_file = os.path.join(work_dir, f"pairs_concept{concept_id}.json")
    with open(pairs_file, "w") as f:
        json.dump(pairs_to_json_doc(pairs, task.get_name()), f)

    activations_file = os.path.join(work_dir, f"activations_concept{concept_id}.json")
    cli.execute_get_activations(_make_args(
        pairs_file=pairs_file, model=model_name, output=activations_file,
        layers=str(layer), extraction_strategy=get_optimal_extraction_strategy(),
        device=args.device, verbose=False, timing=False, raw=False,
        cached_model=model,
    ))

    method_params = {}
    params_file = getattr(args, "method_params_file", None)
    if params_file:
        with open(params_file) as f:
            method_params = json.load(f)
        if not isinstance(method_params, dict):
            raise ValueError(f"--method-params-file {params_file} must hold a JSON object.")

    steering_file = os.path.join(work_dir, f"steering_concept{concept_id}.pt")
    cli.execute_create_steering_object(_make_args(
        enriched_pairs_file=activations_file, output=steering_file,
        method=args.method, verbose=False, timing=False, **method_params,
    ))
    return steering_file, concept


def _generate_steered(
    model,
    instructions: List[str],
    steering_object,
    factor: float,
    strategy: str,
    args,
) -> List[str]:
    """Generate one steered response per instruction at the given factor."""
    messages = [[{"role": "user", "content": instr}] for instr in instructions]
    gen_kwargs: dict = {"max_new_tokens": args.max_new_tokens}
    if args.temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
    return model.generate(
        inputs=messages,
        steering_object=steering_object,
        steering_strength=factor,
        steering_strategy=strategy,
        **gen_kwargs,
    )


def run_concept_steering(
    task,
    concept_id: int,
    model,
    model_name: str,
    args,
    judge,
    work_dir: str,
) -> Dict[str, object]:
    """Run the full AxBench steering protocol for one concept."""
    from wisent.core.control.steering_methods.steering_object import load_steering_object

    layer = args.layer if args.layer is not None else model.num_layers // 2
    steering_file, concept = train_concept_steering_object(
        task, concept_id, model, model_name, layer, args, work_dir,
    )
    steering_object = load_steering_object(steering_file)

    instructions = load_alpaca_instructions(args.n_instructions, args.seed)
    select_set, eval_set = split_select_eval(instructions)
    factors = [float(x) for x in str(args.factors).split(",")]
    if not factors:
        raise ValueError("At least one steering factor is required.")
    strategy = get_optimal("steering_strategy")

    sweep: List[Dict[str, object]] = []
    for factor in factors:
        print(f"   concept {concept_id}: factor {factor} on {len(select_set)} select instructions", flush=True)
        texts = _generate_steered(model, select_set, steering_object, factor, strategy, args)
        scores = score_generations(texts, select_set, concept, judge)
        sweep.append({
            "factor": factor,
            "mean_overall": scores["mean_overall"],
            "mean_concept": scores["mean_concept"],
            "mean_instruct": scores["mean_instruct"],
            "mean_fluency": scores["mean_fluency"],
        })

    best = max(sweep, key=lambda row: row["mean_overall"])
    best_factor = float(best["factor"])
    print(f"   concept {concept_id}: best factor {best_factor} (select overall {best['mean_overall']:.3f})", flush=True)

    eval_texts = _generate_steered(model, eval_set, steering_object, best_factor, strategy, args)
    eval_scores = score_generations(eval_texts, eval_set, concept, judge)

    responses_file = os.path.join(work_dir, f"responses_concept{concept_id}.json")
    with open(responses_file, "w") as f:
        json.dump({
            "task": task.get_name(),
            "model": model_name,
            "concept": concept,
            "concept_id": concept_id,
            "best_factor": best_factor,
            "responses": [
                {
                    "prompt": eval_set[i],
                    "generated_response": eval_texts[i],
                    "metadata": {"concept": concept, "concept_id": concept_id},
                }
                for i in range(len(eval_set))
            ],
        }, f, indent=JSON_INDENT)

    return {
        "concept_id": concept_id,
        "concept": concept,
        "layer": layer,
        "method": args.method,
        "best_factor": best_factor,
        "overall": eval_scores["mean_overall"],
        "concept_score": eval_scores["mean_concept"],
        "instruct_score": eval_scores["mean_instruct"],
        "fluency_score": eval_scores["mean_fluency"],
        "factor_sweep": sweep,
        "responses_file": responses_file,
        "steering_object": steering_file,
    }
