"""
High-level collect-and-cache functions.

Orchestrate activation collection and caching in one step.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from wisent.core.primitives.model_interface.core.activations import (
    ExtractionComponent,
    ExtractionStrategy,
)
from wisent.core.utils.config_tools.constants import (
    RECURSION_INITIAL_DEPTH,
    HASH_PREFIX_LEN,
)
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from .cached_activations import CachedActivations, get_strategy_text_family
from .raw_cached_activations import RawCachedActivations
from .disk_caches import ActivationCache, RawActivationCache

if TYPE_CHECKING:
    from wisent.core.primitives.models.wisent_model import WisentModel


def get_exact_revision_identity(model: object) -> tuple[str | None, str | None]:
    """Return a complete resolved commit pair, or an unpinned identity."""
    revisions = (
        getattr(model, "resolved_model_revision", None),
        getattr(model, "resolved_tokenizer_revision", None),
    )
    if all(
        isinstance(revision, str)
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
        for revision in revisions
    ):
        return revisions
    return None, None


def collect_and_cache_activations(
    model: "WisentModel",
    pairs: List[ContrastivePair],
    benchmark: str,
    strategy: ExtractionStrategy,
    report_interval: int,
    cache: Optional[ActivationCache] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
    component: ExtractionComponent = ExtractionComponent.default(),
    *,
    architecture_module_limit: int,
) -> CachedActivations:
    """
    Collect activations for all pairs and all layers, then cache.

    Args:
        model: WisentModel instance
        pairs: List of contrastive pairs
        benchmark: Benchmark name
        strategy: Extraction strategy
        cache: Optional existing cache to use
        cache_dir: Cache directory (used if cache not provided)
        show_progress: Print progress
        component: Transformer activation component included in the cache identity

    Returns:
        CachedActivations with all layers for all pairs
    """
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector

    model_revision, tokenizer_revision = get_exact_revision_identity(model)
    if cache is None and cache_dir:
        cache = ActivationCache(cache_dir, hash_digest_prefix=HASH_PREFIX_LEN)

    if cache and cache.has(
        model.model_name,
        benchmark,
        strategy,
        component,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    ):
        if show_progress:
            print(f"Loading cached activations for {benchmark}/{strategy.value}")
        loaded = cache.get(
            model.model_name,
            benchmark,
            strategy,
            component,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        if loaded is None:
            raise RuntimeError("activation cache disappeared after a successful cache lookup")
        return loaded

    collector = ActivationCollector(model=model, architecture_module_limit=architecture_module_limit)

    cached = CachedActivations(
        benchmark=benchmark,
        strategy=strategy,
        model_name=model.model_name,
        num_layers=model.num_layers,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )
    for i, pair in enumerate(pairs):
        if show_progress and i % report_interval == RECURSION_INITIAL_DEPTH:
            print(f"Collecting activations: {i+1}/{len(pairs)}", end="\r", flush=True)

        updated = collector.collect(
            pair,
            strategy=strategy,
            layers=None,
            component=component,
        )
        cached.add_pair(
            updated.positive_response.layers_activations,
            updated.negative_response.layers_activations,
        )

    if show_progress:
        print(f"Collected activations: {len(pairs)}/{len(pairs)} pairs, {cached.num_layers} layers")

    if cache:
        cache.put(cached, component)
        if show_progress:
            print(f"Cached to {cache.cache_dir}")

    return cached


def collect_and_cache_raw_activations(
    model: "WisentModel",
    pairs: List[ContrastivePair],
    benchmark: str,
    strategy: ExtractionStrategy,
    report_interval: int,
    cache: Optional[RawActivationCache] = None,
    cache_dir: Optional[str] = None,
    show_progress: bool = True,
    component: ExtractionComponent = ExtractionComponent.default(),
    *,
    architecture_module_limit: int,
) -> RawCachedActivations:
    """Collect and cache exact raw sequences for a pinned model/tokenizer."""
    if not isinstance(component, ExtractionComponent):
        raise TypeError("raw extraction component must be an ExtractionComponent")
    if component.needs_cache:
        raise ValueError(
            "raw collection does not support cache-backed components; "
            "use the standard activation collector for KV_CACHE"
        )
    if component is not ExtractionComponent.RESIDUAL_STREAM:
        raise ValueError(
            "raw collection supports only the residual_stream component"
        )
    from wisent.core.primitives.model_interface.core.activations.activations_collector import ActivationCollector

    text_family = get_strategy_text_family(strategy)
    model_revision, tokenizer_revision = get_exact_revision_identity(model)
    if model_revision is None or tokenizer_revision is None:
        raise ValueError(
            "raw activation collection requires exact resolved model and "
            "tokenizer revisions"
        )
    if cache is None and cache_dir:
        cache = RawActivationCache(
            cache_dir,
            hash_digest_prefix=HASH_PREFIX_LEN,
        )

    if cache and cache.has(
        model.model_name,
        benchmark,
        text_family,
        component,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    ):
        if show_progress:
            print(f"Loading cached raw activations for {benchmark}/{text_family}")
        loaded = cache.get(
            model.model_name,
            benchmark,
            text_family,
            component,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        if loaded is None:
            raise RuntimeError("raw cache disappeared after a successful cache lookup")
        return loaded

    collector = ActivationCollector(
        model=model,
        architecture_module_limit=architecture_module_limit,
    )
    cached = RawCachedActivations(
        benchmark=benchmark,
        text_family=text_family,
        model_name=model.model_name,
        num_layers=model.num_layers,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )

    for index, pair in enumerate(pairs):
        if show_progress and index % report_interval == RECURSION_INITIAL_DEPTH:
            print(
                f"Collecting raw activations: {index + 1}/{len(pairs)}",
                end="\r",
                flush=True,
            )
        metadata = pair.metadata
        if not isinstance(metadata, dict) or "pair_id" not in metadata:
            raise ValueError("raw collection requires pair.metadata['pair_id']")
        raw_data = collector.collect_raw(
            pair,
            strategy=strategy,
            layers=None,
            component=component,
            pair_id=metadata["pair_id"],
            stable_id=pair.stable_id(),
        )
        cached.add_pair(
            pair_id=raw_data["pair_id"],
            stable_id=raw_data["stable_id"],
            pos_hidden_states=raw_data["pos_hidden_states"],
            neg_hidden_states=raw_data["neg_hidden_states"],
            pos_answer_text=raw_data["pos_answer_text"],
            neg_answer_text=raw_data["neg_answer_text"],
            pos_input_ids=raw_data["pos_input_ids"],
            neg_input_ids=raw_data["neg_input_ids"],
            pos_attention_mask=raw_data["pos_attention_mask"],
            neg_attention_mask=raw_data["neg_attention_mask"],
            pos_effective_length=raw_data["pos_effective_length"],
            neg_effective_length=raw_data["neg_effective_length"],
            pos_answer_onset=raw_data["pos_answer_onset"],
            neg_answer_onset=raw_data["neg_answer_onset"],
            pos_answer_end_exclusive=raw_data["pos_answer_end_exclusive"],
            neg_answer_end_exclusive=raw_data["neg_answer_end_exclusive"],
        )

    if show_progress:
        print(
            f"Collected raw activations: {len(pairs)}/{len(pairs)} pairs, "
            f"{cached.num_layers} layers"
        )
    if cache:
        cache.put(cached, component)
        if show_progress:
            print(f"Cached raw to {cache.cache_dir}")
    return cached
