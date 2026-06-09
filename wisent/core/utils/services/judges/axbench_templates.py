"""AxBench judge rubric templates and rating parsing.

The three templates are copied VERBATIM from stanfordnlp/axbench
axbench/evaluators/prompt_templates.py (commit 41c8332). parse_rating is a
line-for-line port of LMJudgeEvaluator._get_rating_from_completion /
_get_ratings_from_completions from axbench/evaluators/lm_judge.py at the same
commit. Do not paraphrase the rubric text: any wording change alters judge
behaviour and breaks comparability with published AxBench numbers.
"""

from __future__ import annotations

__all__ = [
    "CONCEPT_RELEVANCE_TEMPLATE",
    "INSTRUCTION_RELEVANCE_TEMPLATE",
    "FLUENCY_TEMPLATE",
    "parse_rating",
    "RATING_MIN",
    "RATING_MAX",
    "DEFAULT_RATING",
]

RATING_MIN = 0.0
RATING_MAX = 2.0
DEFAULT_RATING = 0.0

CONCEPT_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

INSTRUCTION_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the \
sentence fragment provided below is related to the instruction. Focus solely \
on the degree of relatedness in terms of topic, regardless of grammar, coherence, or \
informativeness.

Begin your evaluation by providing a brief explanation of whether the \
sentence is related to the instruction, and point out references \
related to the instruction. After providing your explanation, rate the instruction \
relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the \
instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, \
and 2 indicates it is more clearly and directly related to the instruction. Provide your rating \
using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""

FLUENCY_TEMPLATE = """[System]
Please act as an impartial judge and evaluate the fluency of the \
sentence fragment provided below. Focus solely on fluency, disregarding \
its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any \
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that \
may hinder readability. After providing your explanation, rate the sentence's fluency \
on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural \
(e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable \
errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. \
Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


def parse_rating(
    completion: str,
    min_rating: float = RATING_MIN,
    max_rating: float = RATING_MAX,
) -> float:
    """Extract the 0/1/2 rating from a judge completion (AxBench semantics).

    Port of LMJudgeEvaluator: split on "Rating:", take the first line, strip
    brackets/quotes/asterisks/trailing period, parse a float. A missing,
    unparseable, or out-of-range rating yields DEFAULT_RATING (0.0) — this is
    the benchmark's own rule, not an error condition.
    """
    if "Rating:" not in completion:
        return DEFAULT_RATING
    rating_text = completion.split("Rating:")[-1].strip()
    rating_text = rating_text.split("\n")[0].strip()
    rating_text = rating_text.replace("[", "").replace("]", "")
    rating_text = rating_text.rstrip(".").strip('"').strip("'").strip("*").strip()
    try:
        rating = float(rating_text)
    except ValueError:
        return DEFAULT_RATING
    if min_rating <= rating <= max_rating:
        return rating
    return DEFAULT_RATING
