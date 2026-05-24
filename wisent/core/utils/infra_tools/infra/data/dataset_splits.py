"""
Unified dataset splitting utilities.

This module provides consistent train/test splitting across all benchmarks,
regardless of their original split structure. All data is pooled together
and split using our own deterministic 80/20 split.
"""

import hashlib
import random
from typing import Any, Dict, List, Tuple

from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, HASH_DISPLAY_LENGTH

DEFAULT_SEED = DEFAULT_RANDOM_SEED


def get_all_docs_from_task(task: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract ALL documents from an lm-eval task, combining all available splits.

    Args:
        task: An lm-eval task object

    Returns:
        Tuple of (all_docs, split_counts) where split_counts shows how many
        docs came from each original split
    """
    all_docs = []
    split_counts = {}

    split_methods = [
        ("training_docs", "has_training_docs"),
        ("validation_docs", "has_validation_docs"),
        ("test_docs", "has_test_docs"),
        ("fewshot_docs", "has_fewshot_docs"),
    ]

    import time
    import random
    for docs_method, has_method in split_methods:
        if not hasattr(task, has_method):
            continue
        # Retry transients: the wisent fleet's 20+ parallel agents
        # saturate HF's 1000-req-per-5-min ceiling. Jittered backoff so
        # 20 agents that all 429'd at the same instant don't all wake at
        # the same instant for the next attempt — that just re-saturates.
        for attempt in range(8):
            try:
                has_docs = getattr(task, has_method)
                if callable(has_docs) and has_docs():
                    docs_iter = getattr(task, docs_method)()
                    if docs_iter is not None:
                        docs = list(docs_iter)
                        if docs:
                            split_counts[docs_method] = len(docs)
                            all_docs.extend(docs)
                break
            except Exception as exc:
                msg = str(exc)
                lower = msg.lower()
                is_transient = (
                    "429" in msg
                    or "too many requests" in lower
                    or "rate limit" in lower
                    or "couldn't find cache" in lower
                    or "couldn't reach" in lower
                    or "connection" in lower and ("timed out" in lower or "reset" in lower)
                )
                if is_transient and attempt < 7:
                    # When the error is "Couldn't find cache for <repo> for config
                    # <X>. Available configs in the cache: [<Y>]", retrying does
                    # nothing — the cache state itself is wrong. Nuke the cache
                    # directory for that dataset so the next attempt fetches all
                    # configs from HF. Confirmed live on 2026-05-06: hundreds of
                    # OALL/Arabic_MMLU jobs failed with stale-cache state where
                    # only one config was downloaded but a different one was
                    # requested by the task.
                    if "couldn't find cache" in lower:
                        import os as _os, re as _re, shutil as _shutil
                        m = _re.search(r"for ([\w\-./]+) for config", msg)
                        if m:
                            ds = m.group(1).replace("/", "___")
                            cache_root = _os.environ.get(
                                "HF_DATASETS_CACHE",
                                _os.path.expanduser("~/.cache/huggingface/datasets"),
                            )
                            for sub in (ds, ds.lower()):
                                p = _os.path.join(cache_root, sub)
                                if _os.path.isdir(p):
                                    _shutil.rmtree(p, ignore_errors=True)
                    base = 30 * (2 ** min(attempt, 5))  # 30,60,120,240,480,960,960,960
                    jitter = random.uniform(0, base)
                    time.sleep(min(base + jitter, 600))
                    continue
                break

    return all_docs, split_counts


def create_deterministic_split(
    all_docs: List[Any],
    benchmark_name: str,
    *,
    train_ratio: float,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Any], List[Any]]:
    """
    Create a deterministic train/test split from all documents.

    Uses benchmark name + seed to create reproducible shuffling,
    ensuring the same split every time for the same benchmark.

    Args:
        all_docs: All documents from the benchmark
        benchmark_name: Name of the benchmark (used for deterministic seeding)
        train_ratio: Ratio of data for training
        seed: Base random seed (default DEFAULT_RANDOM_SEED)

    Returns:
        Tuple of (train_docs, test_docs)
    """
    if not all_docs:
        return [], []

    n = len(all_docs)

    # Create deterministic seed based on benchmark name
    combined_seed = int(hashlib.md5(
        f"{benchmark_name}_{seed}".encode()
    ).hexdigest()[:HASH_DISPLAY_LENGTH], 16)

    # Shuffle indices deterministically
    rng = random.Random(combined_seed)
    indices = list(range(n))
    rng.shuffle(indices)

    # Split
    n_train = int(n * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_docs = [all_docs[i] for i in train_indices]
    test_docs = [all_docs[i] for i in test_indices]

    return train_docs, test_docs


def get_train_docs(
    task: Any,
    benchmark_name: str,
    *,
    train_ratio: float,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Get training documents from a task using our custom split.

    This combines all available splits and returns only the training portion.
    Use this for contrastive pair generation.

    Args:
        task: An lm-eval task object
        benchmark_name: Name of the benchmark (used for deterministic seeding)
        train_ratio: Ratio of data for training
        seed: Random seed for reproducibility

    Returns:
        List of training documents
    """
    all_docs, _ = get_all_docs_from_task(task)
    train_docs, _ = create_deterministic_split(all_docs, benchmark_name, train_ratio=train_ratio, seed=seed)

    return train_docs


def get_test_docs(
    task: Any,
    benchmark_name: str,
    *,
    train_ratio: float,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """
    Get test documents from a task using our custom split.

    This combines all available splits and returns only the test portion.
    Use this for evaluation.

    Args:
        task: An lm-eval task object
        benchmark_name: Name of the benchmark (used for deterministic seeding)
        train_ratio: Ratio of data for training
        seed: Random seed for reproducibility

    Returns:
        List of test documents
    """
    all_docs, _ = get_all_docs_from_task(task)
    _, test_docs = create_deterministic_split(all_docs, benchmark_name, train_ratio=train_ratio, seed=seed)

    return test_docs


def get_split_info(
    task: Any,
    benchmark_name: str,
    *,
    train_ratio: float,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Get information about the split for a task.

    Args:
        task: An lm-eval task object
        benchmark_name: Name of the benchmark (used for deterministic seeding)
        train_ratio: Ratio of data for training
        seed: Random seed

    Returns:
        Dictionary with split information
    """
    all_docs, original_splits = get_all_docs_from_task(task)
    train_docs, test_docs = create_deterministic_split(all_docs, benchmark_name, train_ratio=train_ratio, seed=seed)

    return {
        "benchmark_name": benchmark_name,
        "total_samples": len(all_docs),
        "train_samples": len(train_docs),
        "test_samples": len(test_docs),
        "train_ratio": train_ratio,
        "seed": seed,
        "original_splits": original_splits,
    }
