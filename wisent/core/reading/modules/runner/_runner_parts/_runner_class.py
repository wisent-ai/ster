"""GeometryRunner class for orchestrating geometry search."""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from wisent.core.reading.modules import GeometrySearchSpace
from wisent.core.primitives.model_interface.core.activations import (
    ExtractionComponent,
    ExtractionStrategy,
)
from wisent.core.utils.config_tools.constants import (
    ARCHITECTURE_MODULE_LIMIT_DEFAULT,
    DEFAULT_RANDOM_SEED,
    HASH_PREFIX_LEN,
)
from wisent.core.primitives.model_interface.core.activations.activation_cache import (
    ActivationCache,
    CachedActivations,
    RawActivationCache,
    collect_and_cache_activations,
    collect_and_cache_raw_activations,
    get_strategy_text_family,
)
from wisent.core.primitives.model_interface.core.activations.pipeline.cache.collection import (
    get_exact_revision_identity,
)
from wisent.core.utils import get_layer_combinations
from wisent.core.reading.modules import (
    compute_geometry_metrics,
    generate_nonsense_activations,
)
from wisent.core.reading.modules.runner._runner_parts._test_result import GeometryTestResult
from wisent.core.reading.modules.runner._runner_parts._search_results import GeometrySearchResults


class GeometryRunner:
    """
    Runs geometry search across the search space.

    Uses activation caching for efficiency:
    1. Extract ALL layers once per (benchmark, strategy)
    2. Test all layer combinations from cache
    3. Compare against nonsense baseline (random tokens)
    """

    def __init__(
        self,
        search_space: GeometrySearchSpace,
        model: "WisentModel",
        report_interval: int,
        cache_dir: Optional[str] = None,
        *, train_ratio: float,
    ):
        self.search_space = search_space
        self.model = model
        self.report_interval = report_interval
        self.train_ratio = train_ratio
        self.cache_dir = cache_dir or (
            f"/tmp/wisent_geometry_cache_{model.model_name.replace('/', '_')}"
        )
        self.cache = ActivationCache(
            self.cache_dir,
            hash_digest_prefix=HASH_PREFIX_LEN,
        )
        model_revision, tokenizer_revision = get_exact_revision_identity(model)
        self.raw_cache = (
            RawActivationCache(
                self.cache_dir,
                hash_digest_prefix=HASH_PREFIX_LEN,
            )
            if model_revision is not None and tokenizer_revision is not None
            else None
        )
        self._nonsense_cache: Dict[
            Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _get_nonsense_cache_path(self, n_pairs: int, layer: int) -> Path:
        """Get disk cache path for nonsense baseline."""
        cache_dir = Path(self.cache_dir) / "nonsense_baseline"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = self.model.model_name.replace("/", "_")
        return cache_dir / f"{model_prefix}_n{n_pairs}_layer{layer}.pt"

    def get_nonsense_baseline(
        self,
        n_pairs: int,
        layer: int,
        device: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get or generate nonsense baseline activations.

        Caches results both in memory and on disk so we only generate once
        per (n_pairs, layer) combination.
        """
        cache_key = (n_pairs, layer)
        if cache_key in self._nonsense_cache:
            return self._nonsense_cache[cache_key]
        cache_path = self._get_nonsense_cache_path(n_pairs, layer)
        if cache_path.exists():
            try:
                cached = torch.load(
                    cache_path, map_location="cpu", weights_only=True
                )
                nonsense_pos = cached["positive"]
                nonsense_neg = cached["negative"]
                self._nonsense_cache[cache_key] = (nonsense_pos, nonsense_neg)
                return nonsense_pos, nonsense_neg
            except Exception:
                pass
        device = device or str(self.model.hf_model.device)
        nonsense_pos, nonsense_neg = generate_nonsense_activations(
            model=self.model.hf_model,
            tokenizer=self.model.tokenizer,
            device=device,
            n_pairs=n_pairs,
            layer=layer,
        )
        self._nonsense_cache[cache_key] = (nonsense_pos, nonsense_neg)
        try:
            torch.save({
                "positive": nonsense_pos.cpu(),
                "negative": nonsense_neg.cpu(),
                "n_pairs": n_pairs,
                "layer": layer,
                "model": self.model.model_name,
            }, cache_path)
        except Exception:
            pass
        return nonsense_pos, nonsense_neg

    def clear_nonsense_cache(self, disk: bool = False) -> None:
        """Clear the nonsense baseline cache."""
        self._nonsense_cache.clear()
        if disk:
            cache_dir = Path(self.cache_dir) / "nonsense_baseline"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)

    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        strategies: Optional[List[ExtractionStrategy]] = None,
        max_layer_combo_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> GeometrySearchResults:
        """Run the geometry search."""
        benchmarks = benchmarks or self.search_space.benchmarks
        strategies = strategies or self.search_space.strategies
        max_combo = max_layer_combo_size or self.search_space.config.max_layer_combo_size
        num_layers = self.model.num_layers
        layer_combos = get_layer_combinations(num_layers, max_combo)
        results = GeometrySearchResults(
            model_name=self.model.model_name,
            config=self.search_space.config,
        )
        start_time = time.time()
        extraction_time = 0.0
        test_time = 0.0
        total_extractions = len(benchmarks) * len(strategies)
        extraction_count = 0
        for benchmark in benchmarks:
            for strategy in strategies:
                extraction_count += 1
                if show_progress:
                    print(
                        f"\n[{extraction_count}/{total_extractions}] "
                        f"{benchmark} / {strategy.value}"
                    )
                extract_start = time.time()
                try:
                    cached = self._get_cached_activations(
                        benchmark, strategy, show_progress
                    )
                except Exception as e:
                    if show_progress:
                        print(f"  SKIP: {e}")
                    continue
                extraction_time += time.time() - extract_start
                test_start = time.time()
                for combo in layer_combos:
                    result = compute_geometry_metrics(cached, combo)
                    results.add_result(result)
                test_time += time.time() - test_start
                results.benchmarks_tested = len(
                    set(r.benchmark for r in results.results)
                )
                results.strategies_tested = len(
                    set(r.strategy for r in results.results)
                )
                results.layer_combos_tested = len(results.results)
                if show_progress:
                    print(f"  Tested {len(layer_combos)} layer combos")
        results.total_time_seconds = time.time() - start_time
        results.extraction_time_seconds = extraction_time
        results.test_time_seconds = test_time
        return results

    def _get_cached_activations(
        self,
        benchmark: str,
        strategy: ExtractionStrategy,
        show_progress: bool = True,
    ) -> CachedActivations:
        """Get cached activations, extracting if necessary."""
        component = ExtractionComponent.default()
        model_revision, tokenizer_revision = get_exact_revision_identity(self.model)
        cache_identity = {
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
        }
        if self.cache.has(
            self.model.model_name,
            benchmark,
            strategy,
            component,
            **cache_identity,
        ):
            if show_progress:
                print("  Loading from cache...")
            loaded = self.cache.get(
                self.model.model_name,
                benchmark,
                strategy,
                component,
                **cache_identity,
            )
            if loaded is None:
                raise RuntimeError("activation cache disappeared after lookup")
            return loaded

        text_family = get_strategy_text_family(strategy)
        raw_cache = (
            self.raw_cache
            if model_revision is not None and tokenizer_revision is not None
            else None
        )
        if raw_cache is not None and raw_cache.has(
            self.model.model_name,
            benchmark,
            text_family,
            component,
            **cache_identity,
        ):
            if show_progress:
                print(
                    f"  Loading from raw cache ({text_family} family)..."
                )
            raw_cached = raw_cache.get(
                self.model.model_name,
                benchmark,
                text_family,
                component,
                **cache_identity,
            )
            if raw_cached is None:
                raise RuntimeError("raw activation cache disappeared after lookup")
            cached = raw_cached.to_cached_activations(
                strategy, self.model.tokenizer
            )
            self.cache.put(cached, component)
            return cached

        if show_progress:
            print("  Loading pairs...")
        pairs = self._load_pairs(benchmark)
        if raw_cache is None:
            if show_progress:
                print(f"  Extracting activations for {len(pairs)} pairs...")
            return collect_and_cache_activations(
                model=self.model,
                pairs=pairs,
                benchmark=benchmark,
                strategy=strategy,
                report_interval=self.report_interval,
                cache=self.cache,
                show_progress=show_progress,
                component=component,
                architecture_module_limit=ARCHITECTURE_MODULE_LIMIT_DEFAULT,
            )

        if show_progress:
            print(
                f"  Extracting raw activations for {len(pairs)} pairs "
                f"({text_family} family)..."
            )
        raw_cached = collect_and_cache_raw_activations(
            model=self.model,
            pairs=pairs,
            benchmark=benchmark,
            strategy=strategy,
            report_interval=self.report_interval,
            cache=raw_cache,
            show_progress=show_progress,
            component=component,
            architecture_module_limit=ARCHITECTURE_MODULE_LIMIT_DEFAULT,
        )
        cached = raw_cached.to_cached_activations(
            strategy, self.model.tokenizer
        )
        self.cache.put(cached, component)
        return cached

    def _load_pairs(self, benchmark: str) -> List:
        """Load contrastive pairs for a benchmark."""
        from lm_eval.tasks import TaskManager
        from wisent.extractors.lm_eval.lm_task_pairs_generation import (
            lm_build_contrastive_pairs,
        )
        tm = TaskManager()
        try:
            task_dict = tm.load_task_or_group([benchmark])
            task = list(task_dict.values())[0]
        except Exception:
            task = None
        limit: Optional[int] = self.search_space.config.pairs_per_benchmark
        pairs = lm_build_contrastive_pairs(benchmark, task, limit=limit, train_ratio=self.train_ratio)
        if limit and len(pairs) > limit:
            random.seed(DEFAULT_RANDOM_SEED)
            pairs = random.sample(pairs, limit)
        return pairs
