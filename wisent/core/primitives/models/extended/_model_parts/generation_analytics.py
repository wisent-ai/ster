"""WisentModel method: generate_with_stats."""
from __future__ import annotations

from typing import Any

import torch

from wisent.core.primitives.models.core.atoms import SteeringPlan, GenerationStats, TopLogits
from wisent.core.control.generation.prompts.core.atom import ChatMessage

from wisent.core.primitives.models.config import get_generate_kwargs
from wisent.core.primitives.models.extended._helpers.generation_stopping import build_degeneration_stopping_criteria
from wisent.core.utils.config_tools.constants import AXIS_COLS


@torch.inference_mode()
def _generate_with_stats(
    self,
    inputs: list[list[ChatMessage]],
    steering_strength: float,
    num_return_sequences: int = 1,
    *,
    collect_topk: int,
    use_steering: bool = False,
    steering_plan: SteeringPlan | None = None,
    steering_object: "BaseSteeringObject | None" = None,
    enable_thinking: bool = False,
    ensure_varied_responses: bool = False,
    phrase_ledger: Any = None,
    **gen_kwargs: Any,
) -> tuple[list[str], list[GenerationStats]]:
    """
    Generate with efficient per-token stats (logits / probs), compatible with steering.

    Sampling parameters come from the user's InferenceConfig by default.
    Pass overrides via ``**gen_kwargs``.

    attributes:
        inputs:
            list of chat messages (each a list of {'role','content'} dicts).
        num_return_sequences:
            number of completions to generate per input.
        collect_topk:
            if > 0, collect top-k logits/probs per step for analysis/visualization.
        use_steering:
            if True, apply the current steering plan (if any).
        steering_plan:
            optional SteeringPlan to use for this call only.
        enable_thinking:
            If False, disable thinking/reasoning mode.
        **gen_kwargs:
            overrides for generation kwargs (temperature, max_new_tokens, etc.).

    returns:
        - list of generated strings (length = len(inputs) * num_return_sequences).
        - list of GenerationStats (length = len(inputs) * num_return_sequences).
    """
    # Build generation kwargs from user config + caller overrides
    resolved = get_generate_kwargs(**gen_kwargs)

    if steering_object is not None:
        self.apply_steering_object(steering_object, base_strength=steering_strength)
    elif use_steering:
        self.apply_steering(steering_plan)

    batch = self._batch_encode(inputs, add_generation_prompt=True, enable_thinking=enable_thinking)

    # Build generation kwargs
    generation_kwargs = dict(
        **batch,
        **resolved,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=True,
        output_scores=True,
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

    out = self.hf_model.generate(**generation_kwargs)

    if use_steering or steering_object is not None:
        self.detach()

    texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

    # Extract only assistant responses when using chat templates
    texts = [self._extract_assistant_response(text) for text in texts]

    scores: list[torch.Tensor] = list(out.scores or [])
    stats: list[GenerationStats] = []

    if scores:
        stacked = torch.stack(scores, dim=0)             # [steps, B*num_ret, V]
        steps = stacked.size(0)
        gen_token_ids = out.sequences[:, -steps:]        # [B*num_ret, steps]

        logprobs = torch.log_softmax(stacked.float(), dim=-1)  # [steps, B, V]
        B = logprobs.size(1)
        V = logprobs.size(2)

        for b in range(B):
            toks = gen_token_ids[b].tolist()
            per_step: list[TopLogits] = []
            for t, tok_id in enumerate(toks):
                lp_row = logprobs[t, b]                        # [V]
                logit = scores[t][b, tok_id].item()
                prob = float(lp_row[tok_id].exp().item())
                if collect_topk > 0:
                    topk_vals, topk_ids = lp_row.topk(min(collect_topk, V))
                    per_step.append(TopLogits(
                        token_id=int(tok_id),
                        logit=float(logit),
                        prob=float(prob),
                        topk_ids=topk_ids.tolist(),
                        topk_probs=topk_vals.exp().tolist(),
                    ))
                else:
                    per_step.append(TopLogits(
                        token_id=int(tok_id),
                        logit=float(logit),
                        prob=float(prob),
                    ))
            stats.append(GenerationStats(tokens=toks, per_step=per_step))
    else:
        for _ in range(out.sequences.size(0)):
            stats.append(GenerationStats(tokens=[], per_step=None))

    return texts, stats
