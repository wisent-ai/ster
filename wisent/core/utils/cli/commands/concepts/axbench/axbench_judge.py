"""AxBench judge scoring: three rubric axes + harmonic-mean aggregation.

Single source of truth for the published AxBench aggregation, shared by the
`wisent axbench` command and the evaluate-responses axbench_judge evaluator:
each generation is rated 0/1/2 on concept incorporation, instruction
relatedness and fluency; the overall score is the harmonic mean of the three
with the hard-zero rule (any zero subscore zeroes the overall score).
"""

from typing import Dict, List

from wisent.core.utils.services.judges.base import BaseJudge
from wisent.core.utils.services.judges.axbench_templates import (
    CONCEPT_RELEVANCE_TEMPLATE,
    FLUENCY_TEMPLATE,
    INSTRUCTION_RELEVANCE_TEMPLATE,
    parse_rating,
)

__all__ = ["harmonic_mean", "score_generations"]


def harmonic_mean(scores: List[float]) -> float:
    """AxBench overall score: harmonic mean with the hard-zero rule."""
    if 0 in scores:
        return 0.0
    return len(scores) / sum(1.0 / s for s in scores)


def _mean(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty score list.")
    return sum(values) / len(values)


def score_generations(
    generations: List[str],
    instructions: List[str],
    concept_label: str,
    judge: BaseJudge,
) -> Dict[str, object]:
    """Score steered generations with the verbatim AxBench rubrics.

    arguments:
        generations: steered model outputs, one per instruction.
        instructions: the instructions the generations respond to.
        concept_label: the natural-language concept being steered toward.
        judge: judge client constructed by make_judge.

    returns:
        dict with per-item concept/instruct/fluency/overall score lists and
        their means.
    """
    if len(generations) != len(instructions):
        raise ValueError(
            f"Got {len(generations)} generations for {len(instructions)} "
            "instructions; they must align one-to-one."
        )
    if not generations:
        raise ValueError("No generations to score.")
    if not concept_label:
        raise ValueError("A non-empty concept label is required for judging.")

    concept_prompts = [
        CONCEPT_RELEVANCE_TEMPLATE.format(concept=concept_label, sentence=text)
        for text in generations
    ]
    instruct_prompts = [
        INSTRUCTION_RELEVANCE_TEMPLATE.format(instruction=instruction, sentence=text)
        for instruction, text in zip(instructions, generations)
    ]
    fluency_prompts = [FLUENCY_TEMPLATE.format(sentence=text) for text in generations]

    concept_scores = [parse_rating(c) for c in judge.complete_batch(concept_prompts)]
    instruct_scores = [parse_rating(c) for c in judge.complete_batch(instruct_prompts)]
    fluency_scores = [parse_rating(c) for c in judge.complete_batch(fluency_prompts)]

    overall_scores = [
        harmonic_mean([concept_scores[i], instruct_scores[i], fluency_scores[i]])
        for i in range(len(generations))
    ]

    return {
        "concept_scores": concept_scores,
        "instruct_scores": instruct_scores,
        "fluency_scores": fluency_scores,
        "overall_scores": overall_scores,
        "mean_concept": _mean(concept_scores),
        "mean_instruct": _mean(instruct_scores),
        "mean_fluency": _mean(fluency_scores),
        "mean_overall": _mean(overall_scores),
    }
