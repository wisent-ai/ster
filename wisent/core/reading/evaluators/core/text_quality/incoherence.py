"""
Incoherence detection and composite quality scoring.

Split from gibberish.py only to meet the 300-line file limit; logically the
two halves are one text-quality service. evaluate_quality is the 1-100
composite score that rolls gibberish and incoherence into a single number.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch

from wisent.core.utils.config_tools.constants import (
    SCORE_SCALE_100,
)
from .gibberish import (
    _is_gibberish,
    _get_tokenizer,
    _is_nonsense_word,
)


def _is_incoherent(text: str, min_sentence_length: int, *, nonsense_min_tokens: int) -> bool:
    """
    Detect if text is semantically incoherent or unhelpful.

    Catches issues that pass gibberish detection but are still low quality:
    - Too short to be helpful (< 20 chars for a response)
    - Repeated phrases/sentences
    - Circular/meaningless statements
    - Refusals or non-answers disguised as responses
    """
    if not text:
        return True

    text = text.strip()

    # Check 1: Too short to be a helpful response
    if len(text) < min_sentence_length:
        return True

    # Check 2: Single word or very few words (unhelpful)
    tokens = text.split()
    if len(tokens) < 4:
        return True

    # Check 3: Consecutive duplicate words (e.g., "policymakers policymakers")
    tokens_lower = [t.lower().strip('.,!?"\'-') for t in tokens]
    for i in range(len(tokens_lower) - 1):
        if tokens_lower[i] == tokens_lower[i + 1] and len(tokens_lower[i]) > 2:
            return True

    # Check 4: Repeated sentences - split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        unique_sentences = set(sentences)
        # If more than half the sentences are duplicates, it's repetitive
        if len(unique_sentences) < len(sentences) * 0.5:
            return True

    # Check 5: Repeated phrases (3+ word sequences appearing multiple times)
    if len(tokens) >= 6:
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        trigram_counts = Counter(trigrams)
        most_common_count = trigram_counts.most_common(1)[0][1] if trigrams else 0
        # If any trigram appears 3+ times, it's repetitive
        if most_common_count >= 3:
            return True

    # Check 6: Circular statements that don't add information
    # e.g., "Football, football, and the beautiful game are intertwined, intertwined, intertwined"
    unique_tokens = set(t.lower().strip('.,!?"\'-') for t in tokens)
    # If unique words are less than 40% of total words, very repetitive
    if len(tokens) >= 5 and len(unique_tokens) / len(tokens) < 0.4:
        return True

    # Check 7: Non-answer patterns
    non_answer_patterns = [
        r'^"?no"?\.?$',  # Just "No" or "No."
        r'^therefore\s+there\s+(are|is)\s+no',  # "Therefore there are no..."
        r'^you\s+didn\'?t\s+tell\s+me',  # "You didn't tell me..."
        r'^i\'?ve?\s+got\s+a\s+few\s+of\s+you',  # Nonsensical "I've got a few of you"
    ]
    text_lower = text.lower()
    for pattern in non_answer_patterns:
        if re.match(pattern, text_lower):
            return True

    # Check 8: Excessive repetition of the same word (3+ times for content words)
    content_words = [t.lower().strip('.,!?"\'-') for t in tokens if len(t) >= 4]
    if content_words:
        word_counts = Counter(content_words)
        for word, count in word_counts.items():
            # Skip common filler words
            if word in {'that', 'this', 'have', 'been', 'with', 'from', 'they', 'would', 'could', 'should'}:
                continue
            # If any content word appears more than 3 times in a short response, flag it
            if count >= 3 and count / len(content_words) > 0.15:
                return True

    # Check 9: Nonsense words (using tokenizer fragmentation)
    tokenizer = _get_tokenizer()
    if tokenizer and len(tokens) >= nonsense_min_tokens:
        nonsense_count = 0
        for token in tokens_lower:
            if len(token) >= 4 and _is_nonsense_word(token, tokenizer, nonsense_min_tokens=nonsense_min_tokens):
                nonsense_count += 1
        # If more than 15% of words are nonsense, flag it
        if nonsense_count / len(tokens) > 0.15:
            return True

    return False


def evaluate_quality(
    response: "str | list[str]",
    min_sentence_length: int,
    model: "PreTrainedModel | None" = None,
    tokenizer: "PreTrainedTokenizer | None" = None,
    device: "torch.device | None" = None,
    *,
    nonsense_min_tokens: int,
    quality_min_response_length: int,
    quality_repetition_ratio_threshold: float,
    quality_bigram_repeat_threshold: int,
    quality_bigram_repeat_penalty: float,
    quality_special_char_ratio_threshold: float,
    quality_special_char_penalty: float,
    quality_char_repeat_count: int,
    quality_char_repeat_penalty: float,
) -> float:
    """
    Evaluate response quality using heuristic checks on a scale of 1-100.

    Checks for common quality issues:
    - Gibberish/nonsensical text (immediate zero)
    - Empty or too short responses
    - Repetitive tokens (single token appearing too frequently)
    - Repeated phrases (same bigram appearing multiple times)
    - Nonsensical patterns (excessive special characters)
    - Character repetition (same character repeated many times)

    Args:
        response: The response to evaluate (string or list of strings)
        model: The model (not used, kept for API compatibility)
        tokenizer: The tokenizer (not used, kept for API compatibility)
        device: Device (not used, kept for API compatibility)

    Returns:
        Quality score between 1 and 100
        - 100 = Perfect quality (no issues detected)
        - 1 = Very poor quality (multiple severe issues)
        - 0 = Gibberish detected
    """
    # Handle list inputs - compute average quality
    if isinstance(response, list):
        if not response:
            return 50.0  # Default if empty
        scores = [evaluate_quality(
            r, min_sentence_length=min_sentence_length, nonsense_min_tokens=nonsense_min_tokens,
            quality_min_response_length=quality_min_response_length,
            quality_repetition_ratio_threshold=quality_repetition_ratio_threshold,
            quality_bigram_repeat_threshold=quality_bigram_repeat_threshold,
            quality_bigram_repeat_penalty=quality_bigram_repeat_penalty,
            quality_special_char_ratio_threshold=quality_special_char_ratio_threshold,
            quality_special_char_penalty=quality_special_char_penalty,
            quality_char_repeat_count=quality_char_repeat_count,
            quality_char_repeat_penalty=quality_char_repeat_penalty,
        ) for r in response]
        return sum(scores) / len(scores)

    # Check for gibberish first - immediate zero if detected
    if _is_gibberish(response, nonsense_min_tokens=nonsense_min_tokens):
        return 0.0

    # Check for incoherent/unhelpful responses - immediate zero if detected
    if _is_incoherent(response, min_sentence_length=min_sentence_length, nonsense_min_tokens=nonsense_min_tokens):
        return 0.0

    score = 1.0  # Start with perfect score (0-1 scale)

    # Check 1: Empty or too short
    if len(response.strip()) < quality_min_response_length:
        score *= 0.1
        # Scale to 1-100 and return early
        return max(1.0, score * SCORE_SCALE_100 + 1.0)

    tokens = response.lower().split()

    # Check 2: Repetitive tokens
    if len(tokens) > 0:
        token_counts = Counter(tokens)
        most_common_count = token_counts.most_common(1)[0][1]
        repetition_ratio = most_common_count / len(tokens)

        # Penalize if any token appears more than 30% of the time
        if repetition_ratio > quality_repetition_ratio_threshold:
            score *= 1.0 - (repetition_ratio - quality_repetition_ratio_threshold)

    # Check 3: Repeated n-grams (phrases)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    if bigrams:
        bigram_counts = Counter(bigrams)
        most_common_bigram_count = bigram_counts.most_common(1)[0][1]
        if most_common_bigram_count > quality_bigram_repeat_threshold:
            score *= quality_bigram_repeat_penalty

    # Check 4: Nonsensical patterns (too many special chars)
    special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?']", response)) / max(
        len(response), 1
    )
    if special_char_ratio > quality_special_char_ratio_threshold:
        score *= quality_special_char_penalty

    # Check: Single repeated character
    char_repeat_pattern = rf"(.)\1{{{quality_char_repeat_count},}}"
    if re.search(char_repeat_pattern, response):
        score *= quality_char_repeat_penalty

    # Ensure score is non-negative
    score = max(0.0, score)

    # Scale from 0-1 to 1-100
    quality_score = score * SCORE_SCALE_100 + 1.0

    return quality_score


__all__ = ["_is_incoherent", "evaluate_quality"]
