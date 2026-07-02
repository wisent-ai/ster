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

if TYPE_CHECKING:
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair


def _resp_text(resp_obj: object) -> str:
    for attr in ("model_response", "text"):
        if hasattr(resp_obj, attr) and isinstance(getattr(resp_obj, attr), str):
            return getattr(resp_obj, attr)
    return str(resp_obj)


def collect_raw(
    collector,
    pair: "ContrastivePair",
    strategy: ExtractionStrategy = ExtractionStrategy.default(),
    layers: Sequence[LayerName] | None = None,
    component: ExtractionComponent = ExtractionComponent.default(),
) -> dict:
    """
    Collect RAW hidden states (full sequences) for a contrastive pair.

    Unlike collect(), this returns full hidden state sequences [seq_len, hidden_size]
    instead of extracted single vectors. This allows applying different extraction
    strategies later without re-running the model.

    Args:
        collector: ActivationCollector instance (provides model, store_device)
        pair: The contrastive pair to collect activations for
        strategy: Extraction strategy (determines text construction)
        layers: Which layers to collect, or None for all

    Returns:
        Dict with:
            - pos_hidden_states: {layer_name: [seq_len, hidden_size]}
            - neg_hidden_states: {layer_name: [seq_len, hidden_size]}
            - pos_answer_text: str
            - neg_answer_text: str
            - pos_prompt_len: int
            - neg_prompt_len: int
    """
    pos_text = _resp_text(pair.positive_response)
    neg_text = _resp_text(pair.negative_response)

    needs_other = strategy in (
        ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION
    )
    other_for_pos = neg_text if needs_other else None
    other_for_neg = pos_text if needs_other else None

    pos_data = collect_single_raw(
        collector, pair.prompt, pos_text, strategy, layers,
        other_response=other_for_pos, is_positive=True, component=component
    )
    neg_data = collect_single_raw(
        collector, pair.prompt, neg_text, strategy, layers,
        other_response=other_for_neg, is_positive=False, component=component
    )

    return {
        "pos_hidden_states": pos_data["hidden_states"],
        "neg_hidden_states": neg_data["hidden_states"],
        "pos_answer_text": pos_data["answer_text"],
        "neg_answer_text": neg_data["answer_text"],
        "pos_prompt_len": pos_data["prompt_len"],
        "neg_prompt_len": neg_data["prompt_len"],
    }


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
    """Collect raw hidden states for a single prompt-response pair."""
    collector._ensure_eval_mode()
    with torch.inference_mode():
        tok = collector.model.tokenizer
        full_text, answer_text, prompt_only = build_extraction_texts(
            strategy, prompt, response, tok,
            other_response=other_response, is_positive=is_positive
        )
        if prompt_only:
            prompt_enc = tok(prompt_only, return_tensors="pt", add_special_tokens=False)
            prompt_len = int(prompt_enc["input_ids"].shape[-1])
        else:
            prompt_len = 0

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

        full_enc = tok(
            full_text, return_tensors="pt",
            add_special_tokens=False, truncation=True, max_length=max_len
        )
        compute_device = (
            getattr(collector.model, "compute_device", None)
            or next(collector.model.hf_model.parameters()).device
        )
        full_enc = {k: v.to(compute_device) for k, v in full_enc.items()}

        out = collector.model.hf_model(
            **full_enc, output_hidden_states=True, use_cache=False
        )
        hs = out.hidden_states
        if not hs:
            raise NoHiddenStatesError()

        n_blocks = len(hs) - 1
        names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
        keep = collector._select_indices(layers, n_blocks)

        # For non-residual components, run a second pass with hooks
        hooked = None
        if component.needs_hooks:
            from wisent.core.primitives.model_interface.core.activations.component_hooks import ComponentHookManager
            mgr = ComponentHookManager(collector.model.hf_model, component, keep, collector.architecture_module_limit)
            with mgr.hooks_active():
                collector.model.hf_model(
                    **full_enc, output_hidden_states=False, use_cache=False
                )
            hooked = mgr.get_captured()

        collected: dict[str, torch.Tensor] = {}
        for idx in keep:
            name = names_by_idx[idx]
            if hooked is not None and idx in hooked:
                h = hooked[idx].squeeze(0)
            else:
                h = hs[idx + 1].squeeze(0)
            collected[name] = h.to(collector.store_device)

        return {
            "hidden_states": collected,
            "answer_text": answer_text,
            "prompt_len": prompt_len,
        }
