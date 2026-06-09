"""
Text construction for activation extraction strategies.

Builds the full text input given a strategy, prompt, and response.
Separated from extraction_strategy.py to keep files under 300 lines.
"""

from typing import Tuple, Optional
import warnings

from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_MEDIUM
from .extraction_strategy import ExtractionStrategy, ROLE_PLAY_TOKENS


def _is_qwen_tokenizer(tokenizer) -> bool:
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    cls = type(tokenizer).__name__.lower()
    return "qwen" in name or "qwen" in cls


def _qwen_chat_text(messages, add_generation_prompt: bool = False) -> str:
    lines = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().capitalize() or "User"
        content = str(msg.get("content", ""))
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("Assistant:")
    return "\n".join(lines)


def build_extraction_texts(
    strategy: ExtractionStrategy,
    prompt: str,
    response: str,
    tokenizer,
    other_response: Optional[str] = None,
    is_positive: bool = True,
    auto_convert_strategy: bool = True,
) -> Tuple[str, str, Optional[str]]:
    """
    Build the full text for activation extraction based on strategy.

    Args:
        strategy: The extraction strategy to use
        prompt: The user prompt/question
        response: The response to extract activations for
        tokenizer: The tokenizer (needs apply_chat_template for chat strategies)
        other_response: For mc_balanced/mc_completion, the other response option
        is_positive: For mc_balanced/mc_completion, whether 'response' is the positive option
        auto_convert_strategy: If True, automatically convert strategy to match tokenizer type

    Returns:
        Tuple of (full_text, answer_text, prompt_only_text)
        - full_text: Complete text to feed to model
        - answer_text: The answer portion (for strategies that need it)
        - prompt_only_text: Prompt without answer (for boundary detection)
    """
    if auto_convert_strategy:
        original_strategy = strategy
        strategy = ExtractionStrategy.get_equivalent_for_model_type(strategy, tokenizer)
        if strategy != original_strategy:
            warnings.warn(
                f"Strategy {original_strategy.value} not compatible with tokenizer, "
                f"using {strategy.value} instead.",
                UserWarning
            )

    if strategy in (ExtractionStrategy.CHAT_MEAN, ExtractionStrategy.CHAT_FIRST,
                    ExtractionStrategy.CHAT_LAST, ExtractionStrategy.CHAT_MAX_NORM,
                    ExtractionStrategy.CHAT_WEIGHTED):
        return _build_chat_texts(strategy, prompt, response, tokenizer)

    elif strategy == ExtractionStrategy.ROLE_PLAY:
        return _build_role_play_texts(prompt, response, tokenizer)

    elif strategy == ExtractionStrategy.MC_BALANCED:
        return _build_mc_balanced_texts(
            prompt, response, tokenizer, other_response, is_positive
        )

    elif strategy in (
        ExtractionStrategy.COMPLETION_LAST, ExtractionStrategy.COMPLETION_MEAN
    ):
        prompt_only = f"Q: {prompt}\nA:"
        full_text = f"Q: {prompt}\nA: {response}"
        return full_text, response, prompt_only

    elif strategy == ExtractionStrategy.MC_COMPLETION:
        return _build_mc_completion_texts(
            prompt, response, other_response, is_positive
        )

    else:
        raise ValueError(f"Unknown extraction strategy: {strategy}")


def _build_chat_texts(strategy, prompt, response, tokenizer):
    """Build texts for chat_* strategies."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            f"Strategy {strategy.value} requires a tokenizer with "
            f"apply_chat_template. Base models without chat templates are not "
            f"supported for chat_* strategies. Use completion_last, "
            f"completion_mean, or mc_completion instead."
        )
    if _is_qwen_tokenizer(tokenizer):
        prompt_only = _qwen_chat_text(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        full_text = _qwen_chat_text(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
        )
        return full_text, response, prompt_only

    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return full_text, response, prompt_only


def _build_role_play_texts(prompt, response, tokenizer):
    """Build texts for role_play strategy."""
    random_token = ROLE_PLAY_TOKENS[hash(prompt) % len(ROLE_PLAY_TOKENS)]
    instruction = (
        f"Behave like a person that would answer {prompt} with {response}"
    )

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Strategy role_play requires a tokenizer with "
            "apply_chat_template. Use completion_last or mc_completion "
            "for base models."
        )
    if _is_qwen_tokenizer(tokenizer):
        prompt_only = _qwen_chat_text(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True,
        )
        full_text = _qwen_chat_text(
            [{"role": "user", "content": instruction},
             {"role": "assistant", "content": random_token}],
        )
        return full_text, random_token, prompt_only

    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction},
         {"role": "assistant", "content": random_token}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return full_text, random_token, prompt_only


def _build_mc_balanced_texts(
    prompt, response, tokenizer, other_response, is_positive
):
    """Build texts for mc_balanced strategy."""
    if other_response is None:
        raise ValueError("MC_BALANCED strategy requires other_response")

    pos_goes_in_b = hash(prompt) % 2 == 0

    if is_positive:
        if pos_goes_in_b:
            option_a = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "B"
        else:
            option_a = response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "A"
    else:
        if pos_goes_in_b:
            option_a = response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "A"
        else:
            option_a = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "B"

    mc_prompt = (
        f"Question: {prompt}\n\nWhich is correct?\n"
        f"A. {option_a}\nB. {option_b}\nAnswer:"
    )

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Strategy mc_balanced requires a tokenizer with "
            "apply_chat_template. Use mc_completion for base models."
        )
    if _is_qwen_tokenizer(tokenizer):
        prompt_only = _qwen_chat_text(
            [{"role": "user", "content": mc_prompt}],
            add_generation_prompt=True,
        )
        full_text = _qwen_chat_text(
            [{"role": "user", "content": mc_prompt},
             {"role": "assistant", "content": answer}],
        )
        return full_text, answer, prompt_only

    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": mc_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": mc_prompt},
         {"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return full_text, answer, prompt_only


def _build_mc_completion_texts(
    prompt, response, other_response, is_positive
):
    """Build texts for mc_completion strategy (base models)."""
    if other_response is None:
        raise ValueError("MC_COMPLETION strategy requires other_response")

    pos_goes_in_b = hash(prompt) % 2 == 0

    if is_positive:
        if pos_goes_in_b:
            option_a = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "B"
        else:
            option_a = response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "A"
    else:
        if pos_goes_in_b:
            option_a = response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "A"
        else:
            option_a = other_response[:DISPLAY_TRUNCATION_MEDIUM]
            option_b = response[:DISPLAY_TRUNCATION_MEDIUM]
            answer = "B"

    mc_prompt = (
        f"Question: {prompt}\n\nWhich is correct?\n"
        f"A. {option_a}\nB. {option_b}\nAnswer:"
    )
    prompt_only = mc_prompt
    full_text = f"{mc_prompt} {answer}"
    return full_text, answer, prompt_only
