"""HuggingFace Hub read functions for activation data."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from wisent.core.reading.modules.utilities.data.cache import get_cache_path, save_activations_cache, save_pair_texts_cache
from .hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    activation_hf_path,
    baseline_metadata_hf_path,
    best_method_hf_path,
    model_to_safe_name,
    pair_texts_hf_path,
    test_results_hf_path,
)


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _hf_hub_download(repo_path: str) -> str:
    """Download a file from the HF repo and return local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=repo_path,
        repo_type=HF_REPO_TYPE,
        token=_get_hf_token(),
    )


def _load_safetensors_file(local_path: str) -> tuple:
    """Load tensors and metadata from a safetensors file."""
    from safetensors.torch import load_file
    from safetensors import safe_open

    tensors = load_file(local_path)
    metadata = {}
    with safe_open(local_path, framework="pt") as f:
        metadata = f.metadata() or {}
    return tensors, metadata


def _strict_value_error(code: str, message: str, **details: Any) -> ValueError:
    """Create a stable, machine-readable error for strict data consumers."""
    payload = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return ValueError(json.dumps(payload, sort_keys=True))


def _load_json_strict(path: str) -> Any:
    """Load JSON without silently accepting duplicate object keys."""
    def reject_duplicate_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _strict_value_error(
                    "duplicate_json_key", "JSON object contains a duplicate key", key=key,
                )
            result[key] = value
        return result

    try:
        with open(path, "r") as handle:
            return json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise _strict_value_error(
            "invalid_json", "Required JSON artifact is malformed", path=path, error=str(exc),
        ) from exc
    except OSError as exc:
        raise _strict_value_error(
            "invalid_json", "Could not load required JSON artifact", path=path, error=str(exc),
        ) from exc


def _validate_expected_pair_ids(expected_pair_ids: Sequence[int]) -> List[int]:
    if isinstance(expected_pair_ids, (str, bytes)):
        raise _strict_value_error(
            "invalid_expected_pair_ids", "expected_pair_ids must be an ordered sequence of integers",
        )
    try:
        pair_ids = list(expected_pair_ids)
    except TypeError as exc:
        raise _strict_value_error(
            "invalid_expected_pair_ids", "expected_pair_ids must be an ordered sequence of integers",
        ) from exc
    if not pair_ids:
        raise _strict_value_error("empty_expected_pair_ids", "Expected pair support is empty")
    if any(isinstance(pair_id, bool) or not isinstance(pair_id, int) for pair_id in pair_ids):
        raise _strict_value_error(
            "invalid_pair_id", "Every pair_id must be an integer", pair_ids=pair_ids,
        )
    seen = set()
    duplicates = set()
    for pair_id in pair_ids:
        if pair_id in seen:
            duplicates.add(pair_id)
        seen.add(pair_id)
    duplicates = sorted(duplicates)
    if duplicates:
        raise _strict_value_error(
            "duplicate_expected_pair_id", "Expected pair support contains duplicates",
            duplicate_pair_ids=duplicates,
        )
    return pair_ids


def _manifest_pair_ids(manifest: dict) -> Optional[List[int]]:
    for key in ("pair_ids", "expected_pair_ids", "support_pair_ids"):
        value = manifest.get(key)
        if value is not None:
            return value
    support = manifest.get("support")
    if isinstance(support, dict):
        for key in ("pair_ids", "expected_pair_ids"):
            if support.get(key) is not None:
                return support[key]
    return None


def _validate_completion_manifest(
    manifest: Any,
    *,
    manifest_path: str,
    model_name: str,
    task_name: str,
    layer: int,
    extraction_strategy: str,
    expected_pair_ids: List[int],
) -> List[int]:
    if not isinstance(manifest, dict):
        raise _strict_value_error(
            "invalid_completion_manifest", "Completion manifest must be a JSON object",
            path=manifest_path,
        )
    complete = manifest.get("complete", manifest.get("status"))
    if complete not in (True, "complete", "completed"):
        raise _strict_value_error(
            "incomplete_activation_artifact", "Manifest does not declare the artifact complete",
            path=manifest_path, status=complete,
        )
    manifest_strategy = manifest.get(
        "extraction_strategy", manifest.get("strategy", manifest.get("format")),
    )
    if manifest_strategy != extraction_strategy:
        raise _strict_value_error(
            "extraction_strategy_mismatch", "Completion manifest format does not match the request",
            expected=extraction_strategy, actual=manifest_strategy, path=manifest_path,
        )
    manifest_layers = manifest.get("layers")
    if manifest_layers is None:
        manifest_layers = [manifest.get("layer")] if "layer" in manifest else None
    if manifest_layers != [layer]:
        raise _strict_value_error(
            "layer_mismatch", "Completion manifest layer does not exactly match the request",
            expected=[layer], actual=manifest_layers, path=manifest_path,
        )
    for keys, expected, code in (
        (("model", "model_name"), model_name, "model_mismatch"),
        (("benchmark", "task_name", "task"), task_name, "benchmark_mismatch"),
    ):
        actual = next((manifest[key] for key in keys if key in manifest), None)
        if actual is not None and actual != expected:
            raise _strict_value_error(
                code, "Completion manifest identity does not match the request",
                expected=expected, actual=actual, path=manifest_path,
            )
    manifest_pair_ids = _manifest_pair_ids(manifest)
    if manifest_pair_ids is None:
        raise _strict_value_error(
            "missing_manifest_pair_ids", "Completion manifest does not prove pair support",
            path=manifest_path,
        )
    validated_manifest_ids = _validate_expected_pair_ids(manifest_pair_ids)
    missing_requested_ids = sorted(set(expected_pair_ids).difference(validated_manifest_ids))
    if missing_requested_ids:
        raise _strict_value_error(
            "manifest_support_mismatch", "Completion manifest does not cover requested support",
            missing_pair_ids=missing_requested_ids, path=manifest_path,
        )
    return validated_manifest_ids


def load_activations_from_hf_strict(
    model_name: str,
    task_name: str,
    layer: int,
    extraction_strategy: str,
    expected_pair_ids: Sequence[int],
    completion_manifest_file: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Load one exact, complete activation shard without cache or format fallback.

    Tensors are returned in ``expected_pair_ids`` order. Every validation failure
    is a ``ValueError`` whose string is a JSON object with a stable ``code``.
    """
    allowed_strategies = {
        "chat_first", "chat_last", "chat_mean", "chat_max_norm", "chat_weighted",
        "mc_balanced", "role_play",
    }
    if extraction_strategy not in allowed_strategies:
        raise _strict_value_error(
            "invalid_extraction_strategy", "Strict loading requires an exact supported format",
            extraction_strategy=extraction_strategy, allowed=sorted(allowed_strategies),
        )
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise _strict_value_error(
            "invalid_layer", "Strict loading requires one exact integer layer", layer=layer,
        )
    expected = _validate_expected_pair_ids(expected_pair_ids)
    hf_path = activation_hf_path(model_name, task_name, extraction_strategy, layer)

    if completion_manifest_file is None:
        marker_path = hf_path.rsplit("/", 1)[0] + "/_complete.json"
        try:
            completion_manifest_file = _hf_hub_download(marker_path)
        except Exception as exc:
            raise _strict_value_error(
                "completion_proof_unavailable",
                "Exact activation artifact has no _complete.json proof",
                hf_path=marker_path, error=str(exc),
            ) from exc
    else:
        completion_manifest_file = os.fspath(completion_manifest_file)
    if not Path(completion_manifest_file).is_file():
        raise _strict_value_error(
            "completion_proof_unavailable", "Completion manifest file does not exist",
            path=completion_manifest_file,
        )

    completion_manifest = _load_json_strict(completion_manifest_file)
    manifest_pair_ids = _validate_completion_manifest(
        completion_manifest,
        manifest_path=completion_manifest_file,
        model_name=model_name,
        task_name=task_name,
        layer=layer,
        extraction_strategy=extraction_strategy,
        expected_pair_ids=expected,
    )
    try:
        local_path = _hf_hub_download(hf_path)
    except Exception as exc:
        raise _strict_value_error(
            "activation_artifact_unavailable", "Exact activation artifact is unavailable",
            hf_path=hf_path, error=str(exc),
        ) from exc


    try:
        tensors, metadata = _load_safetensors_file(local_path)
    except Exception as exc:
        raise _strict_value_error(
            "invalid_activation_artifact", "Could not read exact activation artifact",
            hf_path=hf_path, error=str(exc),
        ) from exc
    missing_tensors = sorted(
        {"pos_activations", "neg_activations"}.difference(tensors)
    )
    if missing_tensors:
        raise _strict_value_error(
            "missing_activation_tensor", "Activation artifact is missing required tensors",
            missing_tensors=missing_tensors, hf_path=hf_path,
        )
    pos_tensor = tensors["pos_activations"]
    neg_tensor = tensors["neg_activations"]
    if pos_tensor.ndim < 1 or neg_tensor.ndim < 1 or pos_tensor.shape != neg_tensor.shape:
        raise _strict_value_error(
            "activation_support_mismatch", "Positive and negative tensors must have identical shapes",
            positive_shape=list(pos_tensor.shape), negative_shape=list(neg_tensor.shape),
            hf_path=hf_path,
        )
    if "pair_ids" not in metadata:
        raise _strict_value_error(
            "missing_activation_pair_ids", "Activation artifact metadata has no pair_ids",
            hf_path=hf_path,
        )
    try:
        loaded_pair_ids = _validate_expected_pair_ids(json.loads(metadata["pair_ids"]))
    except (json.JSONDecodeError, TypeError) as exc:
        raise _strict_value_error(
            "invalid_activation_pair_ids", "Activation pair_ids metadata is invalid",
            hf_path=hf_path, error=str(exc),
        ) from exc
    if len(loaded_pair_ids) != pos_tensor.shape[0]:
        raise _strict_value_error(
            "activation_pair_id_count_mismatch", "Tensor rows and pair_ids metadata differ",
            tensor_rows=pos_tensor.shape[0], pair_id_count=len(loaded_pair_ids), hf_path=hf_path,
        )
    if loaded_pair_ids != manifest_pair_ids:
        raise _strict_value_error(
            "manifest_support_mismatch",
            "Activation metadata support or order differs from its completion proof",
            manifest_pair_ids=manifest_pair_ids, activation_pair_ids=loaded_pair_ids,
            hf_path=hf_path,
        )
    row_by_pair_id = {pair_id: index for index, pair_id in enumerate(loaded_pair_ids)}
    ordered_rows = [row_by_pair_id[pair_id] for pair_id in expected]
    return pos_tensor[ordered_rows], neg_tensor[ordered_rows], expected


def load_pair_texts_from_hf_strict(
    task_name: str, expected_pair_ids: Sequence[int],
) -> Dict[int, Dict[str, str]]:
    """Load requested pair texts by explicit or schema-defined pair IDs only."""
    expected = _validate_expected_pair_ids(expected_pair_ids)
    hf_path = pair_texts_hf_path(task_name)
    try:
        local_path = _hf_hub_download(hf_path)
    except Exception as exc:
        raise _strict_value_error(
            "pair_text_artifact_unavailable", "Exact pair-text artifact is unavailable",
            hf_path=hf_path, error=str(exc),
        ) from exc
    raw_data = _load_json_strict(local_path)
    if not isinstance(raw_data, dict):
        raise _strict_value_error(
            "invalid_pair_text_artifact", "Pair-text artifact must be a JSON object",
            hf_path=hf_path,
        )

    raw_pairs = raw_data.get("pairs")
    if raw_pairs is not None:
        if not isinstance(raw_pairs, list):
            raise _strict_value_error(
                "invalid_pair_text_artifact", "Pair-text pairs field must be a list",
                hf_path=hf_path,
            )
        declared_count = raw_data.get("num_pairs")
        if declared_count is not None and (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != len(raw_pairs)
        ):
            raise _strict_value_error(
                "pair_text_count_mismatch", "Declared and actual pair-text counts differ",
                declared_count=declared_count, actual_count=len(raw_pairs), hf_path=hf_path,
            )
        pair_records = [
            (pair.get("pair_id", index) if isinstance(pair, dict) else index, pair)
            for index, pair in enumerate(raw_pairs)
        ]
    else:
        pair_records = list(raw_data.items())

    pairs = {}
    for raw_pair_id, pair in pair_records:
        if isinstance(raw_pair_id, bool) or not isinstance(raw_pair_id, (int, str)):
            raise _strict_value_error(
                "invalid_pair_id", "Pair-text artifact contains a non-integer pair_id",
                pair_id=raw_pair_id, hf_path=hf_path,
            )
        try:
            pair_id = int(raw_pair_id)
        except ValueError as exc:
            raise _strict_value_error(
                "invalid_pair_id", "Pair-text artifact contains a non-integer pair_id",
                pair_id=raw_pair_id, hf_path=hf_path,
            ) from exc
        if isinstance(raw_pair_id, str) and str(pair_id) != raw_pair_id:
            raise _strict_value_error(
                "invalid_pair_id", "Pair-text pair_id is not canonical", pair_id=raw_pair_id,
                hf_path=hf_path,
            )
        if pair_id in pairs:
            raise _strict_value_error(
                "duplicate_pair_id", "Pair-text artifact contains duplicate pair_ids",
                pair_id=pair_id, hf_path=hf_path,
            )
        if not isinstance(pair, dict) or "prompt" not in pair:
            raise _strict_value_error(
                "invalid_pair_text", "Pair text must contain a prompt",
                pair_id=pair_id, hf_path=hf_path,
            )
        positive = pair.get("positive")
        negative = pair.get("negative")
        if positive is None:
            positive_response = pair.get("positive_response")
            if isinstance(positive_response, dict):
                positive = positive_response.get("model_response")
        if negative is None:
            negative_response = pair.get("negative_response")
            if isinstance(negative_response, dict):
                negative = negative_response.get("model_response")
        if any(not isinstance(value, str) for value in (pair["prompt"], positive, negative)):
            raise _strict_value_error(
                "invalid_pair_text", "Pair prompt and responses must be strings",
                pair_id=pair_id, hf_path=hf_path,
            )
        pairs[pair_id] = {
            "prompt": pair["prompt"], "positive": positive, "negative": negative,
        }

    missing_pair_ids = sorted(set(expected).difference(pairs))
    if missing_pair_ids:
        raise _strict_value_error(
            "pair_text_support_mismatch", "Pair-text artifact does not cover requested support",
            missing_pair_ids=missing_pair_ids, hf_path=hf_path,
        )
    return {pair_id: pairs[pair_id] for pair_id in expected}


def load_activations_from_hf(
    model_name: str,
    task_name: str,
    layer: int,
    extraction_strategy: str,
    limit: Optional[int] = None,
    pair_ids: Optional[set] = None,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load activations for a single layer from HuggingFace Hub.

    Downloads the safetensors shard, extracts pos/neg tensors,
    and saves to local cache for future use.

    Args:
        model_name: HuggingFace model ID
        task_name: Benchmark/task name
        layer: Layer number
        extraction_strategy: Extraction strategy used
        limit: Max pairs to return
        pair_ids: Optional set of pair IDs to filter
        use_cache: Whether to use local cache

    Returns:
        Tuple of (pos_activations, neg_activations) tensors
    """
    cache_path = get_cache_path(
        task_name, "activations", model_name=model_name, layer=layer
    )
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        pos_tensor, neg_tensor = cached["pos"], cached["neg"]
        cached_pids = cached.get("pair_ids", list(range(len(pos_tensor))))
        if pair_ids is not None:
            pos_list, neg_list = [], []
            for i, pid in enumerate(cached_pids):
                if pid in pair_ids:
                    pos_list.append(pos_tensor[i])
                    neg_list.append(neg_tensor[i])
            if pos_list:
                return torch.stack(pos_list), torch.stack(neg_list)
        else:
            if limit and len(pos_tensor) > limit:
                return pos_tensor[:limit], neg_tensor[:limit]
            return pos_tensor, neg_tensor

    hf_path = activation_hf_path(
        model_name, task_name, extraction_strategy, layer
    )
    local_path = _hf_hub_download(hf_path)
    tensors, metadata = _load_safetensors_file(local_path)
    pos_tensor = tensors["pos_activations"]
    neg_tensor = tensors["neg_activations"]

    loaded_pair_ids = []
    if "pair_ids" in metadata:
        loaded_pair_ids = json.loads(metadata["pair_ids"])

    if use_cache:
        save_activations_cache(
            task_name, model_name, layer,
            pos_tensor, neg_tensor, loaded_pair_ids,
        )

    if pair_ids is not None:
        pos_list, neg_list = [], []
        for i, pid in enumerate(loaded_pair_ids):
            if pid in pair_ids:
                pos_list.append(pos_tensor[i])
                neg_list.append(neg_tensor[i])
        if pos_list:
            return torch.stack(pos_list), torch.stack(neg_list)
        return torch.tensor([]), torch.tensor([])

    if limit and len(pos_tensor) > limit:
        pos_tensor, neg_tensor = pos_tensor[:limit], neg_tensor[:limit]
    return pos_tensor, neg_tensor


def load_available_layers_from_hf(
    model_name: str,
    task_name: str,
    extraction_strategy: str,
) -> List[int]:
    """Query HF index.json to find available layers."""
    try:
        local_path = _hf_hub_download("index.json")
    except Exception:
        raise FileNotFoundError(
            f"index.json not found in {HF_REPO_ID}. "
            "Repository may not have been initialized yet."
        )

    with open(local_path, "r") as f:
        index = json.load(f)

    safe_model = model_to_safe_name(model_name)
    key = f"{safe_model}/{task_name}/{extraction_strategy}"

    if key not in index:
        raise FileNotFoundError(
            f"No layers found for {model_name}/{task_name}"
            f"/{extraction_strategy} in HF index."
        )
    return sorted(index[key])


def load_pair_texts_from_hf(
    task_name: str,
    limit: int,
    use_cache: bool = True,
) -> Dict[int, Dict[str, str]]:
    """Load contrastive pair texts from HuggingFace Hub.

    Args:
        task_name: Benchmark/task name
        limit: Max pairs to return
        use_cache: Whether to use local cache

    Returns:
        Dict mapping pair_id -> {prompt, positive, negative}
    """
    cache_path = get_cache_path(task_name, "pair_texts")
    if use_cache and cache_path.exists():
        print(f"  Loading pair texts from cache: {cache_path}", flush=True)
        with open(cache_path, "r") as f:
            cached_data = json.load(f)
        pairs = {int(k): v for k, v in cached_data.items()}
        if limit and len(pairs) > limit:
            sorted_ids = sorted(pairs.keys())[:limit]
            pairs = {pid: pairs[pid] for pid in sorted_ids}
        return pairs

    hf_path = pair_texts_hf_path(task_name)
    local_path = _hf_hub_download(hf_path)

    with open(local_path, "r") as f:
        raw_data = json.load(f)

    pairs = {int(k): v for k, v in raw_data.items()}

    if use_cache:
        save_pair_texts_cache(task_name, pairs)

    if limit and len(pairs) > limit:
        sorted_ids = sorted(pairs.keys())[:limit]
        pairs = {pid: pairs[pid] for pid in sorted_ids}

    return pairs


def load_test_results_from_hf(benchmark: str) -> dict | None:
    """Load benchmark test results from HuggingFace Hub.

    Args:
        benchmark: Benchmark/task name.

    Returns:
        Results dict if found, None otherwise.
    """
    hf_path = test_results_hf_path(benchmark)
    try:
        local_path = _hf_hub_download(hf_path)
    except Exception:
        return None

    with open(local_path, "r") as f:
        return json.load(f)


def load_baseline_metadata_from_hf(
    model_name: str, task_name: str,
) -> Optional[dict]:
    """Load baseline metadata (accuracy, total_pairs) from HF Hub."""
    hf_path = baseline_metadata_hf_path(model_name, task_name)
    try:
        local_path = _hf_hub_download(hf_path)
    except Exception:
        return None
    with open(local_path, "r") as f:
        return json.load(f)


def load_best_method_from_hf(
    model_name: str, task_name: str,
) -> Optional[dict]:
    """Load find-best-method results from HF Hub."""
    hf_path = best_method_hf_path(model_name, task_name)
    try:
        local_path = _hf_hub_download(hf_path)
    except Exception:
        return None
    with open(local_path, "r") as f:
        return json.load(f)
