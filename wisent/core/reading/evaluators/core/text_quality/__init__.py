"""
Centralized text-quality checks for evaluators that score free-form model output.

Every evaluator that consumes generated text should gate on this before
scoring. Empty / gibberish / degenerate responses get short-circuited to
UNKNOWN so they don't inflate or deflate the real metric.

Opinionated defaults for the fast-path `check_response_coherence` match the
values that shook out of the Apr 16 truthfulqa_custom steering run, where
gibberish was being rewarded as TRUTHFUL by semantic-similarity scoring.
"""

from __future__ import annotations

from .gibberish import (
    FUNCTION_WORDS,
    _is_gibberish,
    _has_low_function_word_ratio,
    _get_tokenizer,
    _is_nonsense_word,
)
from .incoherence import _is_incoherent, evaluate_quality


# Opinionated defaults for the fast-path coherence gate used by evaluators
# that generate free-form text and do not tune thresholds via optuna.
DEFAULT_NONSENSE_MIN_TOKENS = 6
DEFAULT_MIN_SENTENCE_LENGTH = 20


def check_response_coherence(
    text: str,
    *,
    nonsense_min_tokens: int = DEFAULT_NONSENSE_MIN_TOKENS,
    min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
) -> tuple[bool, str | None]:
    """
    Fast pass/fail coherence gate.

    Returns (is_coherent, reason). `reason` is one of:
        - "empty"     : response was None / empty / whitespace-only
        - "gibberish" : failed _is_gibberish heuristics
        - "incoherent": failed _is_incoherent heuristics
        - None        : response passed, is_coherent is True
    """
    if not text or not text.strip():
        return False, "empty"
    if _is_gibberish(text, nonsense_min_tokens=nonsense_min_tokens):
        return False, "gibberish"
    if _is_incoherent(
        text,
        min_sentence_length=min_sentence_length,
        nonsense_min_tokens=nonsense_min_tokens,
    ):
        return False, "incoherent"
    return True, None


__all__ = [
    "FUNCTION_WORDS",
    "DEFAULT_NONSENSE_MIN_TOKENS",
    "DEFAULT_MIN_SENTENCE_LENGTH",
    "check_response_coherence",
    "evaluate_quality",
    "_is_gibberish",
    "_is_incoherent",
    "_is_nonsense_word",
    "_has_low_function_word_ratio",
    "_get_tokenizer",
]
