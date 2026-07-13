"""
Raw hidden state collection for multi-strategy extraction.

Collects full hidden state sequences [seq_len, hidden_size] instead of
extracted single vectors, allowing applying different extraction strategies
later without re-running the model.

Separated from activations_collector.py to keep files under 300 lines.
"""

from __future__ import annotations
import os
from typing import Sequence, TYPE_CHECKING
import torch

from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerName
from wisent.core.primitives.model_interface.core.activations import (
    ExtractionStrategy,
    ExtractionComponent,
    build_extraction_texts,
)
from wisent.core.utils.infra_tools.errors import NoHiddenStatesError
from wisent.core.primitives.model_interface.core.activations.pipeline.pair_identity import (
    PairId,
    validate_pair_id,
)
from wisent.core.primitives.model_interface.core.activations.pipeline.raw_validation import (
    longest_common_prefix_length,
    require_answer_text,
    require_identity,
    require_supported_raw_component,
    resolved_revision,
    response_text,
)

if TYPE_CHECKING:
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair


def collect_raw(
    collector,
    pair: "ContrastivePair",
    strategy: ExtractionStrategy = ExtractionStrategy.default(),
    layers: Sequence[LayerName] | None = None,
    component: ExtractionComponent = ExtractionComponent.default(),
    *,
    pair_id: PairId,
    stable_id: str,
) -> dict:
    """Collect full-sequence hidden states and exact token metadata."""
    component = require_supported_raw_component(component)
    pair_id = validate_pair_id(pair_id)
    stable_id = require_identity("stable_id", stable_id)
    model_revision = resolved_revision(collector.model, "model")
    tokenizer_revision = resolved_revision(collector.model, "tokenizer")

    pos_text = response_text(pair.positive_response)
    neg_text = response_text(pair.negative_response)
    needs_other = strategy in (
        ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION
    )

    pos_data = collect_single_raw(
        collector=collector,
        prompt=pair.prompt,
        response=pos_text,
        strategy=strategy,
        layers=layers,
        other_response=neg_text if needs_other else None,
        is_positive=True,
        component=component,
    )
    neg_data = collect_single_raw(
        collector=collector,
        prompt=pair.prompt,
        response=neg_text,
        strategy=strategy,
        layers=layers,
        other_response=pos_text if needs_other else None,
        is_positive=False,
        component=component,
    )

    result = {
        "pair_id": pair_id,
        "stable_id": stable_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
    }
    for polarity, data in (("pos", pos_data), ("neg", neg_data)):
        result[f"{polarity}_hidden_states"] = data["hidden_states"]
        result[f"{polarity}_answer_text"] = data["answer_text"]
        result[f"{polarity}_input_ids"] = data["input_ids"]
        result[f"{polarity}_attention_mask"] = data["attention_mask"]
        result[f"{polarity}_effective_length"] = data["effective_length"]
        result[f"{polarity}_answer_onset"] = data["answer_onset"]
        result[f"{polarity}_answer_end_exclusive"] = data["answer_end_exclusive"]
    return result


def collect_single_raw(
    collector,
    prompt: str,
    response: str,
    strategy: ExtractionStrategy,
    layers: Sequence[LayerName] | None,
    other_response: str | None = None,
    is_positive: bool = True,
    component: ExtractionComponent = ExtractionComponent.default(),
) -> dict:
    """Collect one full sequence using a single, offset-bearing tokenization."""
    component = require_supported_raw_component(component)
    response = require_answer_text("response", response)
    if other_response is not None:
        other_response = require_answer_text("other_response", other_response)
    collector._ensure_eval_mode()
    with torch.inference_mode():
        tok = collector.model.tokenizer
        full_text, answer_text, prompt_only = build_extraction_texts(
            strategy,
            prompt,
            response,
            tok,
            other_response=other_response,
            is_positive=is_positive,
        )
        answer_text = require_answer_text("answer_text", answer_text)

        model_cfg = getattr(collector.model.hf_model, "config", None)
        candidates = [
            getattr(tok, "model_max_length", None),
            getattr(model_cfg, "max_position_embeddings", None),
            getattr(model_cfg, "n_positions", None),
            getattr(model_cfg, "max_sequence_length", None),
            getattr(model_cfg, "seq_length", None),
        ]
        sane_limits = []
        for value in candidates:
            try:
                intval = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if 128 < intval < 1_000_000:
                sane_limits.append(intval)
        hard_limit = max(sane_limits) if sane_limits else 4096
        requested = int(os.environ.get("WISENT_RAW_MAX_TOKENS") or hard_limit)
        max_len = max(1, min(requested, hard_limit))

        try:
            full_enc = tok(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )
        except (NotImplementedError, TypeError) as exc:
            raise ValueError(
                "raw collection requires a tokenizer with offset mappings"
            ) from exc
        if "offset_mapping" not in full_enc:
            raise ValueError("tokenizer did not return offset mappings")
        if "input_ids" not in full_enc or "attention_mask" not in full_enc:
            raise ValueError("tokenizer must return input_ids and attention_mask")

        input_batch = torch.as_tensor(full_enc["input_ids"])
        mask_batch = torch.as_tensor(full_enc["attention_mask"])
        offsets_batch = torch.as_tensor(full_enc["offset_mapping"])
        if input_batch.ndim != 2 or input_batch.shape[0] != 1:
            raise ValueError("raw collection expects one encoded input sequence")
        if mask_batch.shape != input_batch.shape:
            raise ValueError("attention mask must match input token IDs")
        if offsets_batch.shape != (*input_batch.shape, 2):
            raise ValueError("offset mapping must contain one character span per token")

        input_ids = input_batch[0].detach().cpu().clone()
        attention_mask = mask_batch[0].detach().cpu().clone()
        offsets = offsets_batch[0].detach().cpu()
        if input_ids.numel() == 0:
            raise ValueError("raw collection produced an empty token sequence")
        if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise ValueError("attention mask must be binary")
        effective_length = int(attention_mask.sum().item())
        if effective_length == 0:
            raise ValueError("raw collection produced no attended tokens")

        prompt_prefix = prompt_only or ""
        search_from = longest_common_prefix_length(prompt_prefix, full_text)
        answer_start = full_text.find(answer_text, search_from)
        if answer_start < 0:
            raise ValueError("answer span cannot be located after the prompt boundary")
        answer_end = answer_start + len(answer_text)
        answer_onset = None
        answer_end_exclusive = None
        for index, ((token_start, token_end), attended) in enumerate(
            zip(offsets.tolist(), attention_mask.tolist())
        ):
            if attended and token_end > answer_start and token_start < answer_end:
                if answer_onset is None:
                    answer_onset = index
                answer_end_exclusive = index + 1
        if answer_onset is None or answer_end_exclusive is None:
            raise ValueError(
                "no attended answer token remains after truncation, or its boundary "
                "cannot be proven from offsets"
            )

        compute_device = (
            getattr(collector.model, "compute_device", None)
            or next(collector.model.hf_model.parameters()).device
        )
        model_inputs = {
            key: value.to(compute_device)
            for key, value in full_enc.items()
            if key != "offset_mapping"
        }
        out = collector.model.hf_model(
            **model_inputs, output_hidden_states=True, use_cache=False
        )
        hs = out.hidden_states
        if not hs:
            raise NoHiddenStatesError()

        n_blocks = len(hs) - 1
        names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
        keep = collector._select_indices(layers, n_blocks)

        hooked = None
        if component.needs_hooks:
            from wisent.core.primitives.model_interface.core.activations.component_hooks import ComponentHookManager
            mgr = ComponentHookManager(
                collector.model.hf_model,
                component,
                keep,
                collector.architecture_module_limit,
            )
            with mgr.hooks_active():
                collector.model.hf_model(
                    **model_inputs, output_hidden_states=False, use_cache=False
                )
            hooked = mgr.get_captured()

        collected: dict[str, torch.Tensor] = {}
        for idx in keep:
            name = names_by_idx[idx]
            if hooked is not None and idx in hooked:
                hidden = hooked[idx].squeeze(0)
            else:
                hidden = hs[idx + 1].squeeze(0)
            if hidden.shape[0] != input_ids.shape[0]:
                raise ValueError("hidden-state sequence length does not match token IDs")
            collected[name] = hidden.to(collector.store_device)

        return {
            "hidden_states": collected,
            "answer_text": answer_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "effective_length": effective_length,
            "answer_onset": answer_onset,
            "answer_end_exclusive": answer_end_exclusive,
        }
