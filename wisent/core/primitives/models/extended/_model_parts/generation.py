"""WisentModel methods: generate, set_steering_from_raw, clear_steering."""
from __future__ import annotations

import logging
from typing import Any

import torch

from wisent.core.primitives.models.core.atoms import SteeringPlan
from wisent.core.primitives.model_interface.core.activations.core.atoms import RawActivationMap
from wisent.core.control.generation.prompts.core.atom import ChatMessage
from wisent.core.primitives.contrastive_pairs.diagnostics import run_control_steering_diagnostics
from wisent.core.utils.infra_tools.errors import ControlVectorDiagnosticsError
from wisent.core.primitives.models.config import get_generate_kwargs
from wisent.core.primitives.models.extended._helpers.generation_stopping import build_degeneration_stopping_criteria
from wisent.core.utils.config_tools.constants import AXIS_COLS, INDEX_FIRST, INDEX_LAST, STEERING_DEFAULT_INTENSITY

logger = logging.getLogger(__name__)


@torch.inference_mode()
def _generate(
    self,
    inputs: list[list[ChatMessage]] | str,
    steering_strength: float = STEERING_DEFAULT_INTENSITY,
    steering_strategy: str = "",
    num_return_sequences: int = 1,
    use_steering: bool = False,
    steering_plan: SteeringPlan | None = None,
    steering_object: "BaseSteeringObject | None" = None,
    steering_strategy_config: dict | None = None,
    enable_thinking: bool = False,
    prompt_is_formatted: bool = False,
    ensure_varied_responses: bool = False,
    phrase_ledger: Any = None,
    **gen_kwargs: Any,
) -> list[str]:
    """
    Batched text generation with optional steering.

    Sampling parameters (temperature, top_p, top_k, max_new_tokens, etc.)
    come from the user's InferenceConfig by default. Pass overrides via
    ``**gen_kwargs`` — e.g. ``model._generate(inputs, do_sample=False)``.

    attributes:
        inputs:
            list of chat messages (each a list of {'role','content'} dicts) OR pre-formatted string.
        num_return_sequences:
            number of completions to generate per input.
        use_steering:
            if True, apply the current steering plan (if any).
        steering_plan:
            optional SteeringPlan to use for this call only.
        steering_object:
            optional SteeringObject for method-specific steering.
        steering_strength:
            Base strength multiplier when using steering_object (default: 1.0).
        enable_thinking:
            If False, disable thinking/reasoning mode.
        prompt_is_formatted:
            If True, inputs is a pre-formatted string with chat template already applied.
        **gen_kwargs:
            overrides for generation kwargs (temperature, max_new_tokens, etc.).

    returns:
        list of generated strings (length = len(inputs) * num_return_sequences).
    """
    # Build generation kwargs from user config + caller overrides
    resolved = get_generate_kwargs(**gen_kwargs)

    if steering_object is not None:
        self.apply_steering_object(
            steering_object,
            base_strength=steering_strength,
            steering_strategy=steering_strategy,
            steering_strategy_config=steering_strategy_config,
            max_new_tokens=resolved["max_new_tokens"],
        )
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
        batch = self._batch_encode(inputs, add_generation_prompt=True, enable_thinking=enable_thinking)

    # Build generation kwargs
    generation_kwargs = dict(
        **batch,
        **resolved,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=False,
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

    # KV cache steering: prefill -> modify cache -> generate from cache
    _ec = getattr(getattr(steering_object, 'metadata', None), 'extraction_component', None)
    if steering_object is not None and _ec == "kv_cache":
        prefill_out = self.hf_model(**batch, use_cache=True)
        past_kv = prefill_out.past_key_values
        _apply_kv_cache_steering(steering_object, past_kv, steering_strength)
        # Generate from modified cache (last token as seed, no hooks)
        cache_gen_kwargs = dict(
            input_ids=batch["input_ids"][:, INDEX_LAST:],
            attention_mask=batch["attention_mask"],
            past_key_values=past_kv,
            use_cache=True,
            **resolved,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
        )
        cache_gen_kwargs["stopping_criteria"] = generation_kwargs["stopping_criteria"]
        if "logits_processor" in generation_kwargs:
            cache_gen_kwargs["logits_processor"] = generation_kwargs["logits_processor"]
        gen_out = self.hf_model.generate(**cache_gen_kwargs)
    else:
        gen_out = self.hf_model.generate(**generation_kwargs)

    if use_steering or steering_object is not None:
        self.detach()

    seqs = gen_out.sequences  # [B * num_return_sequences, T_total]
    texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

    # Extract only assistant responses when using chat templates
    if not prompt_is_formatted and not isinstance(inputs, str):
        texts = [self._extract_assistant_response(text) for text in texts]

    return texts


def _set_steering_from_raw(
    self,
    raw: list[RawActivationMap] | RawActivationMap | None,
    scale: float,
    min_norm_threshold: float,
    layers_description: list[str] | None = None,
    steering_weights: list[float] | None = None,
    normalize: bool = False,
) -> None:
    """
    Replace the internal steering plan using a RawActivationMap (layer_name -> tensor).
    If raw is None or empty, clears any existing steering.
    """
    if not raw:
        self._steering_plan = SteeringPlan()
        return

    # TODO: this should be outside
    reports = run_control_steering_diagnostics(raw, min_norm_threshold=min_norm_threshold)
    for report in reports:
        for issue in report.issues:
            log_method = logger.error if issue.severity == "critical" else logger.warning
            log_method(
                "[control_vector diagnostics] %s (details=%s)",
                issue.message,
                issue.details,
            )

    if any(report.has_critical_issues for report in reports):
        raise ControlVectorDiagnosticsError()

    self._steering_plan = SteeringPlan.from_raw(
        raw=raw,
        layers_description=layers_description,
        weights=steering_weights,
        scale=scale,
        normalize=normalize,
        expected_hidden_size=self._hidden_size
    )


def _clear_steering(self) -> None:
    """
    Remove any existing steering configuration and active hooks.
    After calling this, generation is vanilla.
    """
    self._steering_plan = SteeringPlan()
    self.detach()


def _apply_kv_cache_steering(steering_obj, past_kv, strength: float) -> None:
    """Apply steering vectors to KV cache in-place using standard BaseSteeringObject.

    For each layer in steering_obj.metadata.layers, gets the steering vector,
    reshapes it from flat [num_kv_heads * head_dim] to [num_kv_heads, head_dim]
    using the actual cache shape (GQA-safe), and adds strength * vec to both
    key and value caches at the last sequence position.
    """
    for layer_idx in steering_obj.metadata.layers:
        vec = steering_obj.get_steering_vector(layer_idx)
        # past_kv[layer] = (key_cache, value_cache)
        # cache shape: [batch, num_kv_heads, seq_len, head_dim]
        k_cache = past_kv[layer_idx][INDEX_FIRST]
        v_cache = past_kv[layer_idx][AXIS_COLS]
        num_kv_heads = k_cache.shape[AXIS_COLS]
        head_dim = k_cache.shape[INDEX_LAST]
        vec = vec.to(k_cache.device, k_cache.dtype)
        vec_shaped = vec[:num_kv_heads * head_dim].view(num_kv_heads, head_dim)
        k_cache[:, :, INDEX_LAST, :] += strength * vec_shaped
        v_cache[:, :, INDEX_LAST, :] += strength * vec_shaped
