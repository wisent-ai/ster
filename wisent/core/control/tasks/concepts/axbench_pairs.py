"""Build wisent contrastive pairs from AxBench concept rows.

Pairing strategy: index-zip a concept's positive rows (instruction +
concept-incorporating response) with the shared plain negative rows
(instruction + plain response), truncated to the shorter list. DiffMean/CAA
training depends only on mean(pos) - mean(neg), so the zip order does not
affect the learned direction. Hard negatives, when requested, are placed at
the front of the negative pool (they are the most contrastive responses).
"""

from typing import Any, Dict, List, Optional

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import (
    NegativeResponse,
    PositiveResponse,
)

__all__ = ["build_concept_pairs", "pairs_to_json_doc"]


def build_concept_pairs(
    pos_rows: List[Dict[str, Any]],
    neg_rows: List[Dict[str, Any]],
    concept_id: int,
    concept_label: str,
    hard_neg_rows: Optional[List[Dict[str, Any]]] = None,
    limit: Optional[int] = None,
) -> List[ContrastivePair]:
    """Zip concept positives with negatives into ContrastivePair objects."""
    if not pos_rows:
        raise ValueError(f"No positive rows supplied for concept_id={concept_id}.")
    if not neg_rows and not hard_neg_rows:
        raise ValueError(f"No negative rows supplied for concept_id={concept_id}.")

    negative_pool = list(hard_neg_rows or []) + list(neg_rows)
    n_pairs = min(len(pos_rows), len(negative_pool))
    if limit:
        n_pairs = min(n_pairs, limit)

    pairs: List[ContrastivePair] = []
    for i in range(n_pairs):
        pos = pos_rows[i]
        neg = negative_pool[i]
        pairs.append(
            ContrastivePair(
                prompt=str(pos["input"]),
                positive_response=PositiveResponse(model_response=str(pos["output"])),
                negative_response=NegativeResponse(model_response=str(neg["output"])),
                label=concept_label,
                trait_description=concept_label,
                metadata={"concept_id": concept_id, "concept": concept_label},
            )
        )
    return pairs


def pairs_to_json_doc(pairs: List[ContrastivePair], task_name: str) -> Dict[str, Any]:
    """Serialize pairs into the document shape consumed by get-activations
    and generate-responses --input-file ({'task_name', 'trait_label', 'pairs'})."""
    if not pairs:
        raise ValueError("Cannot serialize an empty pair list.")
    return {
        "task_name": task_name,
        "trait_label": pairs[0].label or task_name,
        "num_pairs": len(pairs),
        "pairs": [pair.to_dict() for pair in pairs],
    }
