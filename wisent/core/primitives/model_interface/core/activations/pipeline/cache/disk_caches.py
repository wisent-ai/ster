"""
Disk-backed caches for activations.

RawActivationCache: full hidden state sequences per (model, benchmark, text_family)
ActivationCache: extracted activation vectors per (model, benchmark, strategy)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

from wisent.core.primitives.model_interface.core.activations import (
    ExtractionComponent,
    ExtractionStrategy,
)
from wisent.core.utils.config_tools.constants import JSON_INDENT
from .cached_activations import (
    CachedActivations,
    get_strategy_text_family,
    validate_revision_identity,
)
from .raw_cached_activations import RawCachedActivations, RawPairData


class RawActivationCache:
    """Disk-backed raw cache isolated by exact model/tokenizer revisions."""

    def __init__(self, cache_dir: str, *, hash_digest_prefix: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, RawCachedActivations] = {}
        self._hash_digest_prefix = hash_digest_prefix

    @staticmethod
    def _validate_revisions(model_revision: str, tokenizer_revision: str) -> None:
        for name, revision in (
            ("model", model_revision),
            ("tokenizer", tokenizer_revision),
        ):
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(char not in "0123456789abcdef" for char in revision)
            ):
                raise ValueError(
                    f"{name} revision must be a lowercase 40-hex commit"
                )

    @staticmethod
    def _component_name(component: object) -> str:
        value = getattr(component, "value", component)
        if not isinstance(value, str) or not value:
            raise ValueError("component must have a non-empty string value")
        if value != ExtractionComponent.RESIDUAL_STREAM.value:
            raise ValueError(
                "raw activation caches support only the residual_stream component"
            )
        return value

    @staticmethod
    def _validate_stored_identity(
        stored: Dict[str, Any],
        *,
        model_name: str,
        benchmark: str,
        text_family: str,
        component: str,
        model_revision: str,
        tokenizer_revision: str,
    ) -> None:
        expected = {
            "schema_version": 2,
            "model_name": model_name,
            "benchmark": benchmark,
            "text_family": text_family,
            "component": component,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
        }
        for field, value in expected.items():
            if stored.get(field) != value:
                raise ValueError(
                    f"raw activation cache {field} does not match requested identity"
                )

    def _get_cache_key(
        self,
        model_name: str,
        benchmark: str,
        text_family: str,
        component: str,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> str:
        self._validate_revisions(model_revision, tokenizer_revision)
        component_name = self._component_name(component)
        key_str = "\x1f".join((
            model_name,
            model_revision,
            tokenizer_revision,
            benchmark,
            text_family,
            component_name,
            "raw-v2",
        ))
        return hashlib.sha256(key_str.encode()).hexdigest()[:self._hash_digest_prefix]

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pt"

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}_meta.json"

    def has(
        self,
        model_name: str,
        benchmark: str,
        text_family: str,
        component: str,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> bool:
        key = self._get_cache_key(
            model_name,
            benchmark,
            text_family,
            component,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        cached = self._memory_cache.get(key)
        if cached is not None:
            if (
                cached.model_name != model_name
                or cached.benchmark != benchmark
                or cached.text_family != text_family
                or cached.model_revision != model_revision
                or cached.tokenizer_revision != tokenizer_revision
            ):
                raise ValueError("raw activation memory cache identity mismatch")
            return True
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return False
        metadata_path = self._get_metadata_path(key)
        if not metadata_path.exists():
            raise ValueError("raw cache metadata sidecar is missing")
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        self._validate_stored_identity(
            metadata,
            model_name=model_name,
            benchmark=benchmark,
            text_family=text_family,
            component=self._component_name(component),
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        return True

    def has_for_strategy(
        self,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        component: str,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> bool:
        return self.has(
            model_name,
            benchmark,
            get_strategy_text_family(strategy),
            component,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )

    def get(
        self,
        model_name: str,
        benchmark: str,
        text_family: str,
        component: str,
        load_to_memory: bool = True,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> Optional[RawCachedActivations]:
        key = self._get_cache_key(
            model_name,
            benchmark,
            text_family,
            component,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        cached = self._memory_cache.get(key)
        if cached is not None:
            if (
                cached.model_name != model_name
                or cached.benchmark != benchmark
                or cached.text_family != text_family
                or cached.model_revision != model_revision
                or cached.tokenizer_revision != tokenizer_revision
            ):
                raise ValueError("raw activation memory cache identity mismatch")
            return cached

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        metadata_path = self._get_metadata_path(key)
        if not metadata_path.exists():
            raise ValueError("raw cache metadata sidecar is missing")
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        component_name = self._component_name(component)
        for stored in (metadata, data):
            self._validate_stored_identity(
                stored,
                model_name=model_name,
                benchmark=benchmark,
                text_family=text_family,
                component=component_name,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
            )

        cached = RawCachedActivations(
            benchmark=data["benchmark"],
            text_family=data["text_family"],
            model_name=data["model_name"],
            num_layers=data["num_layers"],
            model_revision=data["model_revision"],
            tokenizer_revision=data["tokenizer_revision"],
            hidden_size=data["hidden_size"],
        )
        for pair_data in data["pairs"]:
            cached.pairs.append(RawPairData(
                pair_id=pair_data["pair_id"],
                stable_id=pair_data["stable_id"],
                pos_hidden_states=pair_data["pos_hidden_states"],
                neg_hidden_states=pair_data["neg_hidden_states"],
                pos_answer_text=pair_data["pos_answer_text"],
                neg_answer_text=pair_data["neg_answer_text"],
                pos_input_ids=pair_data["pos_input_ids"],
                neg_input_ids=pair_data["neg_input_ids"],
                pos_attention_mask=pair_data["pos_attention_mask"],
                neg_attention_mask=pair_data["neg_attention_mask"],
                pos_effective_length=pair_data["pos_effective_length"],
                neg_effective_length=pair_data["neg_effective_length"],
                pos_answer_onset=pair_data["pos_answer_onset"],
                neg_answer_onset=pair_data["neg_answer_onset"],
                pos_answer_end_exclusive=pair_data["pos_answer_end_exclusive"],
                neg_answer_end_exclusive=pair_data["neg_answer_end_exclusive"],
            ))
        cached.num_pairs = len(cached.pairs)

        if load_to_memory:
            self._memory_cache[key] = cached
        return cached

    def get_for_strategy(
        self,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        component: str,
        load_to_memory: bool = True,
        *,
        model_revision: str,
        tokenizer_revision: str,
    ) -> Optional[RawCachedActivations]:
        return self.get(
            model_name,
            benchmark,
            get_strategy_text_family(strategy),
            component,
            load_to_memory,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )

    def put(
        self,
        cached: RawCachedActivations,
        component: str,
        save_to_disk: bool = True,
    ) -> None:
        key = self._get_cache_key(
            cached.model_name,
            cached.benchmark,
            cached.text_family,
            component,
            model_revision=cached.model_revision,
            tokenizer_revision=cached.tokenizer_revision,
        )
        self._memory_cache[key] = cached
        if not save_to_disk:
            return

        pairs_data = []
        for pair in cached.pairs:
            pairs_data.append({
                "pair_id": pair.pair_id,
                "stable_id": pair.stable_id,
                "pos_hidden_states": pair.pos_hidden_states,
                "neg_hidden_states": pair.neg_hidden_states,
                "pos_answer_text": pair.pos_answer_text,
                "neg_answer_text": pair.neg_answer_text,
                "pos_input_ids": pair.pos_input_ids,
                "neg_input_ids": pair.neg_input_ids,
                "pos_attention_mask": pair.pos_attention_mask,
                "neg_attention_mask": pair.neg_attention_mask,
                "pos_effective_length": pair.pos_effective_length,
                "neg_effective_length": pair.neg_effective_length,
                "pos_answer_onset": pair.pos_answer_onset,
                "neg_answer_onset": pair.neg_answer_onset,
                "pos_answer_end_exclusive": pair.pos_answer_end_exclusive,
                "neg_answer_end_exclusive": pair.neg_answer_end_exclusive,
            })

        data = {
            "schema_version": 2,
            "benchmark": cached.benchmark,
            "text_family": cached.text_family,
            "model_name": cached.model_name,
            "model_revision": cached.model_revision,
            "tokenizer_revision": cached.tokenizer_revision,
            "component": self._component_name(component),
            "num_layers": cached.num_layers,
            "hidden_size": cached.hidden_size,
            "num_pairs": cached.num_pairs,
            "pairs": pairs_data,
        }
        torch.save(data, self._get_cache_path(key))
        metadata = {
            "schema_version": 2,
            "benchmark": cached.benchmark,
            "text_family": cached.text_family,
            "model_name": cached.model_name,
            "model_revision": cached.model_revision,
            "tokenizer_revision": cached.tokenizer_revision,
            "component": self._component_name(component),
            "num_layers": cached.num_layers,
            "hidden_size": cached.hidden_size,
            "num_pairs": cached.num_pairs,
        }
        with open(self._get_metadata_path(key), "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=JSON_INDENT)

    def clear_memory(self) -> None:
        self._memory_cache.clear()


class ActivationCache:
    """Disk-backed extracted activations isolated by exact artifact revisions."""

    def __init__(self, cache_dir: str, *, hash_digest_prefix: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, CachedActivations] = {}
        self._hash_digest_prefix = hash_digest_prefix

    @staticmethod
    def _component_name(component: object) -> str:
        value = getattr(component, "value", component)
        if not isinstance(value, str) or not value:
            raise ValueError("component must have a non-empty string value")
        return value

    def _get_cache_key(
        self,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        component: str,
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> str:
        validate_revision_identity(model_revision, tokenizer_revision)
        component_name = self._component_name(component)
        if model_revision is None:
            # Keep the original namespace readable for genuinely unpinned loads.
            key_str = f"{model_name}_{benchmark}_{strategy.value}"
            if component_name != "residual_stream":
                key_str += f"_{component_name}"
            return hashlib.md5(key_str.encode()).hexdigest()[:self._hash_digest_prefix]
        key_str = "\x1f".join((
            "revision-pinned",
            model_name,
            model_revision,
            tokenizer_revision,
            benchmark,
            strategy.value,
            component_name,
        ))
        return hashlib.sha256(key_str.encode()).hexdigest()[:self._hash_digest_prefix]

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pt"

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    @staticmethod
    def _validate_stored_identity(
        stored: Dict[str, Any],
        *,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        component: str,
        model_revision: str | None,
        tokenizer_revision: str | None,
    ) -> None:
        if (
            stored.get("model_revision") != model_revision
            or stored.get("tokenizer_revision") != tokenizer_revision
        ):
            raise ValueError("activation cache revision does not match requested revision")
        expected = {
            "model_name": model_name,
            "benchmark": benchmark,
            "strategy": strategy.value,
        }
        for field, value in expected.items():
            if stored.get(field) != value:
                raise ValueError(f"activation cache {field} does not match requested identity")
        stored_component = stored.get("component")
        if stored_component is None and model_revision is None:
            return
        if stored_component != component:
            raise ValueError("activation cache component does not match requested identity")

    def has(
        self,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        component: str,
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> bool:
        component_name = self._component_name(component)
        key = self._get_cache_key(
            model_name,
            benchmark,
            strategy,
            component_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        cached = self._memory_cache.get(key)
        if cached is not None:
            validate_revision_identity(cached.model_revision, cached.tokenizer_revision)
            if (
                cached.model_revision != model_revision
                or cached.tokenizer_revision != tokenizer_revision
            ):
                raise ValueError("activation cache revision does not match requested revision")
            return True
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return False
        metadata_path = self._get_metadata_path(key)
        if not metadata_path.exists():
            if model_revision is None:
                return True
            raise ValueError("activation cache metadata sidecar is missing")
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
        self._validate_stored_identity(
            metadata,
            model_name=model_name,
            benchmark=benchmark,
            strategy=strategy,
            component=component_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        return True

    def get(
        self,
        model_name: str,
        benchmark: str,
        strategy: ExtractionStrategy,
        component: str,
        load_to_memory: bool = True,
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> Optional[CachedActivations]:
        component_name = self._component_name(component)
        key = self._get_cache_key(
            model_name,
            benchmark,
            strategy,
            component_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        cached = self._memory_cache.get(key)
        if cached is not None:
            if (
                cached.model_revision != model_revision
                or cached.tokenizer_revision != tokenizer_revision
            ):
                raise ValueError("activation cache revision does not match requested revision")
            return cached

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        metadata_path = self._get_metadata_path(key)
        if not metadata_path.exists() and model_revision is not None:
            raise ValueError("activation cache metadata sidecar is missing")
        if metadata_path.exists():
            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)
            self._validate_stored_identity(
                metadata,
                model_name=model_name,
                benchmark=benchmark,
                strategy=strategy,
                component=component_name,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
            )

        data = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        self._validate_stored_identity(
            data,
            model_name=model_name,
            benchmark=benchmark,
            strategy=strategy,
            component=component_name,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )
        cached = CachedActivations(
            benchmark=data["benchmark"],
            strategy=ExtractionStrategy(data["strategy"]),
            model_name=data["model_name"],
            num_layers=data["num_layers"],
            model_revision=data.get("model_revision"),
            tokenizer_revision=data.get("tokenizer_revision"),
            hidden_size=data["hidden_size"],
        )
        cached.pair_activations = data["pair_activations"]
        cached.num_pairs = data["num_pairs"]
        if load_to_memory:
            self._memory_cache[key] = cached
        return cached

    def put(
        self,
        cached: CachedActivations,
        component: str,
        save_to_disk: bool = True,
    ) -> None:
        validate_revision_identity(cached.model_revision, cached.tokenizer_revision)
        component_name = self._component_name(component)
        key = self._get_cache_key(
            cached.model_name,
            cached.benchmark,
            cached.strategy,
            component_name,
            model_revision=cached.model_revision,
            tokenizer_revision=cached.tokenizer_revision,
        )
        self._memory_cache[key] = cached
        if not save_to_disk:
            return

        identity = {
            "benchmark": cached.benchmark,
            "strategy": cached.strategy.value,
            "model_name": cached.model_name,
            "component": component_name,
            "model_revision": cached.model_revision,
            "tokenizer_revision": cached.tokenizer_revision,
            "num_layers": cached.num_layers,
            "hidden_size": cached.hidden_size,
            "num_pairs": cached.num_pairs,
        }
        data = {**identity, "pair_activations": cached.pair_activations}
        torch.save(data, self._get_cache_path(key))
        with open(self._get_metadata_path(key), "w") as metadata_file:
            json.dump(identity, metadata_file, indent=JSON_INDENT)

    def clear_memory(self) -> None:
        self._memory_cache.clear()

    def list_cached(self) -> List[Dict[str, Any]]:
        result = []
        for meta_path in self.cache_dir.glob("*.json"):
            with open(meta_path) as metadata_file:
                result.append(json.load(metadata_file))
        return result

    def get_cache_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.cache_dir.glob("*.pt"))
