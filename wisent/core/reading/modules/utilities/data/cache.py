"""Local caching for database data."""
import json
from pathlib import Path
import torch
from wisent.core.control.steering_methods.configs.optimal import get_optimal

CACHE_DIR = Path.home() / ".wisent_cache"


def get_cache_path(task_name: str, cache_type: str, **kwargs) -> Path:
    """Get cache file path for a given task and type."""
    component = kwargs.get("component", get_optimal("extraction_component"))
    # Default component uses old path format for backward compatibility
    comp_suffix = "" if component == "residual_stream" else f"_{component}"

    if cache_type == "pair_texts":
        path = CACHE_DIR / f"{task_name}_pair_texts.json"
    elif cache_type == "activations":
        model_name = kwargs.get("model_name", "unknown").replace("/", "_")
        layer = kwargs.get("layer", 0)
        path = CACHE_DIR / f"{task_name}_{model_name}_layer{layer}{comp_suffix}_activations.pt"
    elif cache_type == "viz":
        model_name = kwargs.get("model_name", "unknown").replace("/", "_")
        layer = kwargs.get("layer", 0)
        path = CACHE_DIR / f"{task_name}_{model_name}_layer{layer}_viz.json"
    else:
        path = CACHE_DIR / f"{task_name}_{cache_type}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_pair_texts_cache(task_name: str, limit: int = None):
    """Load pair texts from cache if available.

    Returns None for any cache miss, including corrupt JSON (truncated
    writes from a prior ENOSPC) or read errors. Without this defense the
    JSONDecodeError propagated to the CLI and the whole job failed
    instead of rebuilding from the upstream task. The corrupt file is
    unlinked so the next run can write a clean one.
    """
    cache_path = get_cache_path(task_name, "pair_texts")
    if not cache_path.exists():
        return None

    print(f"  Loading pair texts from cache: {cache_path}")
    try:
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  pair_texts cache unreadable ({exc!r}); removing and rebuilding")
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None
    pairs = {int(k): v for k, v in cached_data.items()}

    if limit and len(pairs) > limit:
        pair_ids = sorted(pairs.keys())[:limit]
        pairs = {pid: pairs[pid] for pid in pair_ids}
    return pairs


def save_pair_texts_cache(task_name: str, pairs: dict):
    """Save pair texts to cache.

    Best-effort: if /home is full (ENOSPC=28) the write is skipped with
    a warning instead of crashing the job. The truncated partial file
    is removed so the next read won't trip JSONDecodeError. The dataset
    is still available from the upstream HF dataset, so a missed cache
    write is recoverable on the next invocation.
    """
    cache_path = get_cache_path(task_name, "pair_texts")
    print(f"  Caching {len(pairs)} pair texts to: {cache_path}")
    try:
        with open(cache_path, 'w') as f:
            json.dump(pairs, f)
    except OSError as exc:
        print(f"  pair_texts cache write failed ({exc!r}); continuing without cache")
        try:
            cache_path.unlink()
        except OSError:
            pass


def load_activations_cache(task_name: str, model_name: str, layer: int, pair_ids=None, limit=None):
    """Load activations from cache if available."""
    cache_path = get_cache_path(task_name, "activations", model_name=model_name, layer=layer)
    if not cache_path.exists():
        return None, None, None

    print(f"  Loading activations from cache: {cache_path}")
    cached = torch.load(cache_path, weights_only=True)
    pos_tensor = cached["pos"]
    neg_tensor = cached["neg"]
    cached_pair_ids = cached.get("pair_ids", list(range(len(pos_tensor))))

    if pair_ids is not None:
        pos_list, neg_list = [], []
        for i, pid in enumerate(cached_pair_ids):
            if pid in pair_ids:
                pos_list.append(pos_tensor[i])
                neg_list.append(neg_tensor[i])
        if pos_list:
            return torch.stack(pos_list), torch.stack(neg_list), None
        return None, None, None

    if limit and len(pos_tensor) > limit:
        pos_tensor = pos_tensor[:limit]
        neg_tensor = neg_tensor[:limit]

    return pos_tensor, neg_tensor, cached_pair_ids


def save_activations_cache(task_name: str, model_name: str, layer: int, pos_tensor, neg_tensor, pair_ids):
    """Save activations to cache."""
    cache_path = get_cache_path(task_name, "activations", model_name=model_name, layer=layer)
    print(f"  Caching {len(pos_tensor)} activations to: {cache_path}")
    torch.save({
        "pos": pos_tensor,
        "neg": neg_tensor,
        "pair_ids": pair_ids,
    }, cache_path)


def get_cached_layers(task_name: str, model_name: str) -> list:
    """Get list of layers available in cache for a task/model combination."""
    model_safe = model_name.replace("/", "_")
    pattern = f"{task_name}_{model_safe}_layer*_activations.pt"

    layers = []
    for cache_file in CACHE_DIR.glob(pattern):
        # Extract layer number from filename
        name = cache_file.stem  # e.g., "truthfulqa_custom_meta-llama_Llama-3.2-1B-Instruct_layer12_activations"
        parts = name.split("_layer")
        if len(parts) == 2:
            layer_part = parts[1].split("_")[0]  # e.g., "12"
            try:
                layers.append(int(layer_part))
            except ValueError:
                pass

    return sorted(layers)


def load_viz_cache(task_name: str, model_name: str, layer: int) -> dict | None:
    """Load cached visualizations from local cache or HF."""
    cache_path = get_cache_path(task_name, "viz", model_name=model_name, layer=layer)
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import viz_cache_hf_path
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import _hf_hub_download
        hf_path = viz_cache_hf_path(model_name, task_name, layer)
        local = _hf_hub_download(hf_path)
        with open(local, "r") as f:
            data = json.load(f)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data
    except Exception:
        return None


def save_viz_cache(
    task_name: str, model_name: str, layer: int, visualizations: dict,
) -> None:
    """Save visualizations to local cache and HF."""
    cache_path = get_cache_path(task_name, "viz", model_name=model_name, layer=layer)
    with open(cache_path, "w") as f:
        json.dump(visualizations, f)
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
            viz_cache_hf_path, HF_REPO_ID, HF_REPO_TYPE,
        )
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import _get_api
        hf_path = viz_cache_hf_path(model_name, task_name, layer)
        _get_api().upload_file(
            path_or_fileobj=str(cache_path), path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
    except Exception:
        pass
