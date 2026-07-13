"""WisentModel method: generate_stream."""
from __future__ import annotations

import threading
from typing import Any, Iterable

import torch
from transformers import TextIteratorStreamer

from wisent.core.primitives.models.core.atoms import SteeringPlan
from wisent.core.control.generation.prompts.core.atom import ChatMessage
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.primitives.models.config import get_generate_kwargs
from wisent.core.primitives.models.extended._helpers.generation_stopping import build_degeneration_stopping_criteria
from wisent.core.utils.config_tools.constants import AXIS_COLS

# Non-blocking join for the generation worker thread
_JOIN_NOWAIT = 0.0


@torch.inference_mode()
def _generate_stream(
    self,
    inputs: list[list[ChatMessage]] | str,
    steering_strength: float,
    use_steering: bool = False,
    steering_plan: SteeringPlan | None = None,
    steering_object: "BaseSteeringObject | None" = None,
    skip_prompt: bool = True,
    skip_special_tokens: bool = True,
    enable_thinking: bool = False,
    prompt_is_formatted: bool = False,
    ensure_varied_responses: bool = False,
    phrase_ledger: Any = None,
    **gen_kwargs: Any,
) -> Iterable[str]:
    """
    Streamed text generation with optional steering.
    Uses the TextIteratorStreamer from transformers.

    Sampling parameters come from the user's InferenceConfig by default.
    Pass overrides via ``**gen_kwargs``.

    attributes:
        inputs:
            list of chat messages (each a list of {'role','content'} dicts) OR pre-formatted string.
            Currently only one conversation is supported.
        use_steering:
            if True, apply the current steering plan (if any).
        steering_plan:
            optional SteeringPlan to use for this call only.
        skip_prompt:
            if True, the yielded text excludes the input prompt.
        skip_special_tokens:
            if True, special tokens are removed from the yielded text.
        enable_thinking:
            If False, disable thinking/reasoning mode.
        prompt_is_formatted:
            If True, inputs is a pre-formatted string with chat template already applied.
        **gen_kwargs:
            overrides for generation kwargs (temperature, max_new_tokens, etc.).

    yields:
        generated text chunks (str), as they become available.
    """

    # Build generation kwargs from user config + caller overrides
    resolved = get_generate_kwargs(**gen_kwargs)

    if steering_object is not None:
        self.apply_steering_object(steering_object, base_strength=steering_strength)
    elif use_steering:
        self.apply_steering(steering_plan)

    if prompt_is_formatted and isinstance(inputs, str):
        # Direct tokenization of pre-formatted prompt
        tokenizer_output = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=False,  # Single prompt, no padding needed
            truncation=True,  # Avoid errors on long inputs
            max_length=self.tokenizer.model_max_length,  # Use model's actual limit
        )
        # Move tensors to the correct device (same as _batch_encode does)
        batch = {
            "input_ids": tokenizer_output["input_ids"].to(self.device),
            "attention_mask": tokenizer_output["attention_mask"].to(self.device)
        }
    else:
        # Current behavior: apply chat template
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise InsufficientDataError(
                reason=f"generate_stream currently supports exactly one conversation at a time (got {type(inputs)} with {len(inputs) if isinstance(inputs, list) else 'N/A'} items)"
            )
        batch = self._batch_encode(inputs, add_generation_prompt=True, enable_thinking=enable_thinking)

    streamer = TextIteratorStreamer(
        self.tokenizer,
        skip_prompt=skip_prompt,
        skip_special_tokens=skip_special_tokens,
    )

    generation_kwargs = dict(
        batch,
        **resolved,
        return_dict_in_generate=False,
        output_scores=False,
        streamer=streamer,
    )

    # Add diversity processors if requested
    if ensure_varied_responses and phrase_ledger:
        from wisent.core.control.tasks.base.diversity_processors import build_diversity_processors
        logits_processors = build_diversity_processors(self.tokenizer, phrase_ledger)
        if logits_processors:
            generation_kwargs['logits_processor'] = logits_processors

    # Add degeneration detection — stops sequences with repetitive n-grams
    # or decoded gibberish (CamelCase salads, low function-word ratio, etc.).
    prompt_length = batch["input_ids"].shape[AXIS_COLS]
    generation_kwargs["stopping_criteria"] = build_degeneration_stopping_criteria(
        prompt_length, self.tokenizer,
    )

    runtime_hooks = getattr(self, "_active_steering_runtime", None)
    if runtime_hooks is not None:
        runtime_hooks.begin_generation()

    worker = threading.Thread(
        target=self.hf_model.generate,
        kwargs=generation_kwargs,
        daemon=True,
    )
    worker.start()

    try:
        for new_text in streamer:
            if new_text:
                yield new_text
    finally:
        if use_steering or steering_object is not None:
            self.detach()
        worker.join(_JOIN_NOWAIT)
