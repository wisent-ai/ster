from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING
import torch

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, LayerName, RawActivationMap
from wisent.core.primitives.model_interface.core.activations import (
    ExtractionStrategy,
    ExtractionComponent,
    build_extraction_texts,
    extract_activation,
)
from wisent.core.utils.infra_tools.errors import NoHiddenStatesError
from wisent.core.utils.config_tools.constants import LOG_EPS, RECURSION_INITIAL_DEPTH

if TYPE_CHECKING:
    from wisent.core.primitives.models.wisent_model import WisentModel

__all__ = ["ActivationCollector"]


@dataclass(slots=True)
class ActivationCollector:
    """
    Collect per-layer activations for contrastive pairs.

    Args:
        model: WisentModel instance
        store_device: Device to store collected activations on (default: "cpu" to avoid GPU OOM)
        dtype: Optional torch.dtype to cast activations to
    """

    model: "WisentModel"
    architecture_module_limit: int
    store_device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None
    _last_qk: dict = field(default_factory=dict, init=False, repr=False)

    def collect(
        self,
        pair: ContrastivePair,
        strategy: ExtractionStrategy = ExtractionStrategy.default(),
        layers: Sequence[LayerName] | None = None,
        normalize: bool = False,
        component: ExtractionComponent = ExtractionComponent.default(),
        aggregation: ExtractionStrategy | None = None,
        return_full_sequence: bool = False,
        normalize_layers: bool = False,
        prompt_strategy: str | None = None,
        capture_qk: bool = False,
    ) -> ContrastivePair:
        """Collect activations for a contrastive pair."""
        if aggregation is not None:
            strategy = aggregation
        if normalize_layers:
            normalize = True
        pos_text = _resp_text(pair.positive_response)
        neg_text = _resp_text(pair.negative_response)
        needs_other = strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION)
        other_for_pos = neg_text if needs_other else None
        other_for_neg = pos_text if needs_other else None
        pos = self._collect_single(
            pair.prompt, pos_text, strategy, layers, normalize,
            other_response=other_for_pos, is_positive=True, component=component,
            capture_qk=capture_qk)
        pos_qk = dict(self._last_qk) if capture_qk else {}
        neg = self._collect_single(
            pair.prompt, neg_text, strategy, layers, normalize,
            other_response=other_for_neg, is_positive=False, component=component,
            capture_qk=capture_qk)
        neg_qk = dict(self._last_qk) if capture_qk else {}
        if capture_qk:
            self._last_qk = {"pos": pos_qk, "neg": neg_qk}
        return pair.with_activations(positive=pos, negative=neg)

    def _collect_single(
        self, prompt: str, response: str,
        strategy: ExtractionStrategy, layers: Sequence[LayerName] | None,
        normalize: bool = False, other_response: str | None = None,
        is_positive: bool = True,
        component: ExtractionComponent = ExtractionComponent.default(),
        capture_qk: bool = False,
    ) -> LayerActivations:
        """Collect activations for a single prompt-response pair."""
        self._ensure_eval_mode()
        with torch.inference_mode():
            tok = self.model.tokenizer
            full_text, answer_text, prompt_only = build_extraction_texts(
                strategy, prompt, response, tok,
                other_response=other_response, is_positive=is_positive)
            if prompt_only:
                prompt_enc = tok(prompt_only, return_tensors="pt", add_special_tokens=False)
                prompt_len = int(prompt_enc["input_ids"].shape[-1])
            else:
                prompt_len = 0
            _capped = min(int(tok.model_max_length or 4096), 4096)
            full_enc = tok(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=_capped)
            compute_device = getattr(self.model, "compute_device", None) or next(self.model.hf_model.parameters()).device
            full_enc = {k: v.to(compute_device) for k, v in full_enc.items()}
            n_blocks = self.model.num_layers
            names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
            keep = self._select_indices(layers, n_blocks)
            if component.needs_cache:
                from wisent.core.primitives.model_interface.core.activations.hooks.kv_cache.kv_cache_collector import collect_kv_cache
                return collect_kv_cache(
                    self.model.hf_model, full_enc, keep, names_by_idx,
                    strategy, answer_text, tok, prompt_len,
                    store_device=self.store_device, dtype=self.dtype)
            qk_mgrs = []
            if capture_qk:
                from wisent.core.primitives.model_interface.core.activations.component_hooks import ComponentHookManager
                for comp in (ExtractionComponent.Q_PROJ, ExtractionComponent.K_PROJ):
                    mgr = ComponentHookManager(self.model.hf_model, comp, keep, self.architecture_module_limit)
                    mgr._register_hooks()
                    qk_mgrs.append(mgr)
            out = self.model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if not hs:
                raise NoHiddenStatesError()
            if qk_mgrs:
                q_cap, k_cap = qk_mgrs[0].get_captured(), qk_mgrs[1].get_captured()
                for m in qk_mgrs:
                    m._remove_hooks()
                qk_out = {}
                for idx in keep:
                    nm = names_by_idx[idx]
                    if idx in q_cap:
                        qk_out[f"q_{nm}"] = extract_activation(
                            strategy, q_cap[idx].squeeze(0), answer_text, tok, prompt_len).to(self.store_device)
                    if idx in k_cap:
                        qk_out[f"k_{nm}"] = extract_activation(
                            strategy, k_cap[idx].squeeze(0), answer_text, tok, prompt_len).to(self.store_device)
                self._last_qk = qk_out
            if component.needs_hooks:
                hooked = self._collect_with_hooks(
                    full_enc, keep, component, strategy, answer_text, tok, prompt_len)
            else:
                hooked = None
            collected: RawActivationMap = {}
            for idx in keep:
                name = names_by_idx[idx]
                if hooked is not None and idx in hooked:
                    h = hooked[idx].squeeze(0)
                else:
                    h = hs[idx + 1].squeeze(0)
                value = extract_activation(strategy, h, answer_text, tok, prompt_len)
                value = value.to(self.store_device)
                if self.dtype is not None:
                    value = value.to(self.dtype)
                if normalize:
                    value = self._normalize(value)
                collected[name] = value
            return LayerActivations(collected)

    def _collect_with_hooks(self, full_enc, keep, component, strategy, answer_text, tok, prompt_len):
        """Run a second forward pass with hooks to capture component activations."""
        from wisent.core.primitives.model_interface.core.activations.component_hooks import ComponentHookManager
        manager = ComponentHookManager(self.model.hf_model, component, keep, self.architecture_module_limit)
        with manager.hooks_active():
            self.model.hf_model(**full_enc, output_hidden_states=False, use_cache=False)
        return manager.get_captured()

    def collect_raw(
        self,
        pair: ContrastivePair,
        strategy: ExtractionStrategy = ExtractionStrategy.default(),
        layers: Sequence[LayerName] | None = None,
        component: ExtractionComponent = ExtractionComponent.default(),
    ) -> dict:
        """Collect RAW hidden states (full sequences) for a contrastive pair."""
        from wisent.core.primitives.model_interface.core.activations.raw_collector import collect_raw
        return collect_raw(self, pair, strategy, layers, component=component)

    def collect_multi_strategy(
        self,
        pair: ContrastivePair,
        strategies: Sequence[ExtractionStrategy],
        layers: Sequence[LayerName] | None = None,
        component: ExtractionComponent = ExtractionComponent.default(),
        capture_qk: bool = False,
    ) -> dict:
        """Run ONE forward pass per side and apply multiple strategy aggregations.

        All strategies in `strategies` MUST share a text family (i.e.
        build_extraction_texts produces the same (full_text, answer_text,
        prompt_only) for all of them given the same prompt/response). The
        canonical use is the chat_* family (chat_last, chat_mean, chat_first,
        chat_max_norm, chat_weighted), where only the post-hoc aggregation of
        hidden states differs.

        Returns:
            {strategy: {"pair": ContrastivePair (with activations),
                        "qk": {"pos": {...}, "neg": {...}}}}
            for every requested strategy. Caller picks the standard collect()
            output shape from "pair" and the same _last_qk-style dict from "qk".

        MC strategies (MC_BALANCED / MC_COMPLETION) need the *other* response
        and produce a strategy-specific input text (the option ordering is
        randomized via hash(prompt) % 2). They are not safe to bundle with
        other strategies — the caller must pass them alone, or as a single
        family group of size 1.
        """
        if not strategies:
            return {}
        strategies = list(strategies)
        needs_other = any(
            s in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION)
            for s in strategies
        )
        if needs_other and len(strategies) > 1:
            raise ValueError(
                "MC_BALANCED / MC_COMPLETION cannot share a forward pass with "
                "other strategies (the option ordering changes the input text)."
            )

        pos_text = _resp_text(pair.positive_response)
        neg_text = _resp_text(pair.negative_response)
        other_for_pos = neg_text if needs_other else None
        other_for_neg = pos_text if needs_other else None

        pos_per_strategy = self._collect_single_multi(
            pair.prompt, pos_text, strategies, layers,
            other_response=other_for_pos, is_positive=True,
            component=component, capture_qk=capture_qk,
        )
        neg_per_strategy = self._collect_single_multi(
            pair.prompt, neg_text, strategies, layers,
            other_response=other_for_neg, is_positive=False,
            component=component, capture_qk=capture_qk,
        )

        out: dict = {}
        for strategy in strategies:
            new_pair = pair.with_activations(
                positive=pos_per_strategy[strategy]["acts"],
                negative=neg_per_strategy[strategy]["acts"],
            )
            qk = {
                "pos": pos_per_strategy[strategy].get("qk", {}),
                "neg": neg_per_strategy[strategy].get("qk", {}),
            }
            out[strategy] = {"pair": new_pair, "qk": qk}
        return out

    def _collect_single_multi(
        self,
        prompt: str,
        response: str,
        strategies: Sequence[ExtractionStrategy],
        layers: Sequence[LayerName] | None,
        other_response: str | None = None,
        is_positive: bool = True,
        component: ExtractionComponent = ExtractionComponent.default(),
        capture_qk: bool = False,
    ) -> dict:
        """Forward pass once; apply multiple strategy aggregations.

        Returns: {strategy: {"acts": LayerActivations, "qk": dict[str, Tensor]}}
        """
        if component.needs_cache:
            raise NotImplementedError(
                "KV cache component cannot share forward passes across strategies"
            )

        self._ensure_eval_mode()
        with torch.inference_mode():
            tok = self.model.tokenizer
            # All strategies in this batch share the same input text by
            # contract (caller groups by text family). Use the first as
            # the build reference.
            ref_strategy = strategies[0]
            full_text, answer_text, prompt_only = build_extraction_texts(
                ref_strategy, prompt, response, tok,
                other_response=other_response, is_positive=is_positive,
            )
            if prompt_only:
                prompt_enc = tok(prompt_only, return_tensors="pt", add_special_tokens=False)
                prompt_len = int(prompt_enc["input_ids"].shape[-1])
            else:
                prompt_len = 0
            _capped = min(int(tok.model_max_length or 4096), 4096)
            full_enc = tok(
                full_text, return_tensors="pt",
                add_special_tokens=False, truncation=True, max_length=_capped,
            )
            compute_device = (
                getattr(self.model, "compute_device", None)
                or next(self.model.hf_model.parameters()).device
            )
            full_enc = {k: v.to(compute_device) for k, v in full_enc.items()}
            n_blocks = self.model.num_layers
            names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
            keep = self._select_indices(layers, n_blocks)

            qk_mgrs = []
            if capture_qk:
                from wisent.core.primitives.model_interface.core.activations.component_hooks import ComponentHookManager
                for comp in (ExtractionComponent.Q_PROJ, ExtractionComponent.K_PROJ):
                    mgr = ComponentHookManager(
                        self.model.hf_model, comp, keep, self.architecture_module_limit,
                    )
                    mgr._register_hooks()
                    qk_mgrs.append(mgr)

            out = self.model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if not hs:
                raise NoHiddenStatesError()

            # Capture once, apply each strategy's aggregation across Q/K.
            qk_per_strategy: dict = {s: {} for s in strategies}
            if qk_mgrs:
                q_cap, k_cap = qk_mgrs[0].get_captured(), qk_mgrs[1].get_captured()
                for m in qk_mgrs:
                    m._remove_hooks()
                for strategy in strategies:
                    qk_out: dict[str, torch.Tensor] = {}
                    for idx in keep:
                        nm = names_by_idx[idx]
                        if idx in q_cap:
                            qk_out[f"q_{nm}"] = extract_activation(
                                strategy, q_cap[idx].squeeze(0), answer_text, tok, prompt_len,
                            ).to(self.store_device)
                        if idx in k_cap:
                            qk_out[f"k_{nm}"] = extract_activation(
                                strategy, k_cap[idx].squeeze(0), answer_text, tok, prompt_len,
                            ).to(self.store_device)
                    qk_per_strategy[strategy] = qk_out

            # Hooked component (e.g. mlp_output) shares one forward pass too;
            # extract_activation is applied per-strategy on the same captured
            # tensors.
            if component.needs_hooks:
                hooked = self._collect_with_hooks(
                    full_enc, keep, component, ref_strategy, answer_text, tok, prompt_len,
                )
            else:
                hooked = None

            result: dict = {}
            for strategy in strategies:
                collected: RawActivationMap = {}
                for idx in keep:
                    name = names_by_idx[idx]
                    if hooked is not None and idx in hooked:
                        h = hooked[idx].squeeze(0)
                    else:
                        h = hs[idx + 1].squeeze(0)
                    value = extract_activation(strategy, h, answer_text, tok, prompt_len)
                    value = value.to(self.store_device)
                    if self.dtype is not None:
                        value = value.to(self.dtype)
                    collected[name] = value
                result[strategy] = {
                    "acts": LayerActivations(collected),
                    "qk": qk_per_strategy.get(strategy, {}),
                }
            return result

    def _collect_single_raw(self, prompt, response, strategy, layers, other_response=None, is_positive=True):
        from wisent.core.primitives.model_interface.core.activations.raw_collector import collect_single_raw
        return collect_single_raw(self, prompt, response, strategy, layers, other_response=other_response, is_positive=is_positive)

    def collect_batched(
        self,
        texts: list[str],
        report_interval: int,
        batch_size: int,
        strategy: ExtractionStrategy = ExtractionStrategy.default(),
        layers: Sequence[LayerName] | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        """Collect activations for multiple texts in batches."""
        self._ensure_eval_mode()
        results: list[dict[str, torch.Tensor]] = []

        tok = self.model.tokenizer
        compute_device = getattr(self.model, "compute_device", None) or next(self.model.hf_model.parameters()).device

        num_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.inference_mode():
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]

                if show_progress and batch_idx % report_interval == RECURSION_INITIAL_DEPTH:
                    print(f"Processing batch {batch_idx + 1}/{num_batches}...", end='\r', flush=True)

                encoded = tok(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=tok.model_max_length, add_special_tokens=True,
                )
                encoded = {k: v.to(compute_device) for k, v in encoded.items()}

                try:
                    out = self.model.hf_model(**encoded, output_hidden_states=True, use_cache=False)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    out = self.model.hf_model(**encoded, output_hidden_states=True, use_cache=False)

                hs = out.hidden_states
                if not hs:
                    raise NoHiddenStatesError()

                n_blocks = len(hs) - 1
                names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
                keep = self._select_indices(layers, n_blocks)
                attention_mask = encoded.get("attention_mask")

                for i in range(len(batch_texts)):
                    collected: dict[str, torch.Tensor] = {}
                    if attention_mask is not None:
                        seq_len = int(attention_mask[i].sum().item())
                    else:
                        seq_len = hs[0].shape[1]

                    for idx in keep:
                        name = names_by_idx[idx]
                        h = hs[idx + 1][i, :seq_len, :]

                        if strategy == ExtractionStrategy.CHAT_MEAN:
                            value = h.mean(dim=0)
                        elif strategy == ExtractionStrategy.CHAT_FIRST:
                            value = h[0]
                        elif strategy in (ExtractionStrategy.CHAT_LAST, ExtractionStrategy.ROLE_PLAY,
                                          ExtractionStrategy.MC_BALANCED,
                                          ExtractionStrategy.CHAT_MAX_NORM, ExtractionStrategy.CHAT_WEIGHTED):
                            value = h[-1]
                        else:
                            raise ValueError(f"Unsupported strategy for batched collection: {strategy}")

                        collected[name] = value.to(self.store_device)

                    results.append(collected)

                del out, hs, encoded
                torch.cuda.empty_cache()

        if show_progress:
            print(f"Processed {len(texts)} texts in {num_batches} batches" + " " * 20)

        return results

    def _select_indices(self, layer_names: Sequence[str] | None, n_blocks: int) -> list[int]:
        """Map layer names '1'..'L' -> indices 0..L-1."""
        if not layer_names:
            return list(range(n_blocks))
        out: list[int] = []
        for name in layer_names:
            try:
                i = int(name)
            except ValueError:
                raise KeyError(f"Layer name must be numeric string like '3', got {name!r}")
            if not (1 <= i <= n_blocks):
                raise IndexError(f"Layer '{i}' out of range 1..{n_blocks}")
            out.append(i - 1)
        return sorted(set(out))

    def _normalize(self, x: torch.Tensor, dim: int = -1, eps: float = LOG_EPS) -> torch.Tensor:
        """Safely L2-normalize tensor."""
        if not torch.is_floating_point(x):
            return x
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
        mask = norm > eps
        safe_norm = torch.where(mask, norm, torch.ones_like(norm))
        y = x / safe_norm
        return torch.where(mask, y, torch.zeros_like(y))

    def _ensure_eval_mode(self) -> None:
        try:
            self.model.hf_model.eval()
        except Exception:
            pass


def _resp_text(resp_obj: object) -> str:
    for attr in ("model_response", "text"):
        if hasattr(resp_obj, attr) and isinstance(getattr(resp_obj, attr), str):
            return getattr(resp_obj, attr)
    return str(resp_obj)
