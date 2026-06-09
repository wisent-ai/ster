"""AxBench concept-detection protocol for one concept.

Per the reference AUCROCEvaluator: score every token representation of each
test-split passage at the chosen layer by dot product with the per-concept
steering vector, max-pool per sequence, min-max normalise over the concept's
evaluation set, and report AUROC plus best-threshold F1 in the balanced
(equal negatives) and imbalanced (extra negatives) settings.
"""

import random
from typing import Dict, List

import torch
from sklearn.metrics import f1_score, roc_auc_score

from wisent.core.utils.cli.commands.concepts.axbench.axbench_steer import (
    train_concept_steering_object,
)

__all__ = ["run_concept_detection"]


@torch.inference_mode()
def _max_activation(model, text: str, layer: int, direction: torch.Tensor) -> float:
    """Max-pooled dot product between token representations and direction."""
    encoded = model.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=model.tokenizer.model_max_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    output = model.hf_model(**encoded, output_hidden_states=True)
    hidden = output.hidden_states[layer][0].float()  # [seq_len, hidden]
    scores = hidden @ direction
    return scores.max().item()


def _min_max_normalize(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        raise ValueError(
            "Detection scores are constant across the evaluation set; "
            "min-max normalisation is undefined."
        )
    return [(v - lo) / (hi - lo) for v in values]


def _best_f1(labels: List[int], scores: List[float]) -> float:
    """Maximise F1 over candidate thresholds (the paper binarises by
    choosing the threshold that maximises F1)."""
    best = 0.0
    for threshold in sorted(set(scores)):
        predictions = [1 if s >= threshold else 0 for s in scores]
        best = max(best, f1_score(labels, predictions))
    return best


def _evaluate_set(
    model, pos_texts: List[str], neg_texts: List[str], layer: int, direction: torch.Tensor,
) -> Dict[str, float]:
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    if len(set(labels)) < 2:
        raise ValueError(
            f"Detection evaluation set is single-class "
            f"({len(pos_texts)} positives, {len(neg_texts)} negatives)."
        )
    raw = [_max_activation(model, text, layer, direction) for text in pos_texts + neg_texts]
    normalized = _min_max_normalize(raw)
    return {
        "auroc": float(roc_auc_score(labels, normalized)),
        "f1": float(_best_f1(labels, normalized)),
        "n_positive": len(pos_texts),
        "n_negative": len(neg_texts),
    }


def run_concept_detection(
    task,
    concept_id: int,
    model,
    model_name: str,
    args,
    work_dir: str,
) -> Dict[str, object]:
    """Run AxBench concept detection for one concept."""
    layer = args.layer if args.layer is not None else model.num_layers // 2
    steering_file, concept = train_concept_steering_object(
        task, concept_id, model, model_name, layer, args, work_dir,
    )
    from wisent.core.control.steering_methods.steering_object import load_steering_object

    steering_object = load_steering_object(steering_file)
    direction = steering_object.get_steering_vector(int(layer)).to(model.device).float()

    rows = task.detection_rows(concept_id)
    pos_texts = [str(row["output"]) for row in rows["positive"]]
    neg_pool = [str(row["output"]) for row in rows["negative"]]
    extra_pool = [str(row["output"]) for row in rows["extra_negative"]]
    if len(neg_pool) < len(pos_texts):
        raise ValueError(
            f"concept_id={concept_id}: only {len(neg_pool)} negatives for "
            f"{len(pos_texts)} positives; cannot build the balanced set."
        )

    rng = random.Random(args.seed)
    balanced_negs = rng.sample(neg_pool, len(pos_texts))
    # Imbalanced setting: the concept's own negatives plus extra plain
    # negatives drawn from other concepts (paper: ~3600 additional).
    extra_count = min(args.imbalanced_negatives, len(extra_pool))
    imbalanced_negs = neg_pool + rng.sample(extra_pool, extra_count)

    print(f"   concept {concept_id}: detection over {len(pos_texts)} positives", flush=True)
    balanced = _evaluate_set(model, pos_texts, balanced_negs, layer, direction)
    imbalanced = _evaluate_set(model, pos_texts, imbalanced_negs, layer, direction)

    return {
        "concept_id": concept_id,
        "concept": concept,
        "layer": layer,
        "method": args.method,
        "auroc": balanced["auroc"],
        "f1": balanced["f1"],
        "balanced": balanced,
        "imbalanced": imbalanced,
        "steering_object": steering_file,
    }
