#!/usr/bin/env python3
"""Materialize immutable activation proofs from an inventory-v2 preflight descriptor."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import time
from typing import Any, Dict, Mapping, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request

try:
    from .desired_results_selection import SelectionError, validate_inventory_selection
    from .desired_results_pair_source import PairSourceError, validate_pair_source
    from .desired_results_target import (
        METHODS, STRATEGIES, canonical_json, canonical_sha256, finalize_target_manifest,
        result_id, target_id,
    )
except ImportError:
    try:
        from scripts.steering.desired_results_selection import SelectionError, validate_inventory_selection
        from scripts.steering.desired_results_pair_source import PairSourceError, validate_pair_source
        from scripts.steering.desired_results_target import (
            METHODS, STRATEGIES, canonical_json, canonical_sha256, finalize_target_manifest,
            result_id, target_id,
        )
    except ImportError:
        from desired_results_selection import SelectionError, validate_inventory_selection  # type: ignore
        from desired_results_pair_source import PairSourceError, validate_pair_source  # type: ignore
        from desired_results_target import (  # type: ignore
            METHODS, STRATEGIES, canonical_json, canonical_sha256, finalize_target_manifest,
            result_id, target_id,
        )

MAX_HEADER_BYTES = 16 * 1024 * 1024
HEADER_PROBE_BYTES = 64 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
RETRIES = 4
ROOT_KEYS = {
    "schema_version", "descriptor_kind", "protocol", "target", "source_evidence",
    "expected_routes", "no_submit", "descriptor_sha256",
}
TARGET_KEYS = {
    "target_id", "result_id", "model_name", "model_slug", "benchmark",
    "expected_pairs", "layer_count",
}
EVIDENCE_KEYS = {
    "inventory_sha256", "activation_cache_sha256", "activation_record_sha256",
    "activation_repo_id", "activation_repo_type", "activation_revision",
    "model_revision", "tokenizer_revision", "observed_n_pairs", "observed_grouped",
    "observed_strategy_layers",
}


class PreflightError(RuntimeError):
    """The immutable source does not satisfy the requested proof contract."""


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PreflightError(f"{label} must have exactly {sorted(keys)}; got {actual}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreflightError(f"{label} must be a non-empty string")
    return value


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PreflightError(f"{label} must be a positive integer")
    return value


def _sha(value: Any, label: str, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise PreflightError(f"{label} must be a lowercase {length}-character hex digest")
    return value


def _read_descriptor(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read descriptor {path}: {exc}") from exc
    root = _exact(value, ROOT_KEYS, "descriptor")
    if root["schema_version"] != 2 or root["descriptor_kind"] != "activation_proof_preflight":
        raise PreflightError("descriptor schema_version/kind is not activation_proof_preflight v2")
    if root["no_submit"] is not True:
        raise PreflightError("descriptor must be explicitly non-submitting")
    supplied_hash = _sha(root["descriptor_sha256"], "descriptor_sha256")
    unhashed = dict(root)
    del unhashed["descriptor_sha256"]
    if canonical_sha256(unhashed) != supplied_hash:
        raise PreflightError("descriptor_sha256 does not match canonical descriptor content")
    protocol = _exact(root["protocol"], {"id", "revision"}, "protocol")
    protocol_id = _string(protocol["id"], "protocol.id")
    _positive(protocol["revision"], "protocol.revision")
    target = _exact(root["target"], TARGET_KEYS, "target")
    for key in ("target_id", "result_id", "model_name", "model_slug", "benchmark"):
        _string(target[key], f"target.{key}")
    model_slug = target["model_slug"]
    if ":" in model_slug or "/" in model_slug or "\x00" in model_slug or model_slug in {".", ".."}:
        raise PreflightError("target.model_slug is not a path-safe identity component")
    benchmark_parts = target["benchmark"].split("/")
    if "\x00" in target["benchmark"] or any(part in {"", ".", ".."} for part in benchmark_parts):
        raise PreflightError("target.benchmark must be a safe non-absolute category path")
    expected_pairs = _positive(target["expected_pairs"], "target.expected_pairs")
    layer_count = _positive(target["layer_count"], "target.layer_count")
    expected_target_id = target_id(protocol_id, model_slug, target["benchmark"])
    expected_result_id = result_id(protocol_id, model_slug, target["benchmark"])
    if target["target_id"] != expected_target_id or target["result_id"] != expected_result_id:
        raise PreflightError("target_id/result_id do not match the shared hashed identity contract")
    evidence = _exact(root["source_evidence"], EVIDENCE_KEYS, "source_evidence")
    for key in ("inventory_sha256", "activation_cache_sha256", "activation_record_sha256"):
        _sha(evidence[key], f"source_evidence.{key}")
    for key in ("activation_revision", "model_revision", "tokenizer_revision"):
        _sha(evidence[key], f"source_evidence.{key}", 40)
    _string(evidence["activation_repo_id"], "source_evidence.activation_repo_id")
    if evidence["activation_repo_type"] not in {"dataset", "model", "space"}:
        raise PreflightError("source_evidence.activation_repo_type is invalid")
    if type(evidence["observed_n_pairs"]) is not int or evidence["observed_n_pairs"] != expected_pairs or evidence["observed_grouped"] is not False:
        raise PreflightError("aggregate activation evidence has wrong pair count or is grouped")
    observed = _exact(evidence["observed_strategy_layers"], set(STRATEGIES), "observed_strategy_layers")
    if any(type(observed[name]) is not int or observed[name] != layer_count for name in STRATEGIES):
        raise PreflightError("aggregate activation evidence does not cover every strategy/layer")
    expected_routes = [
        {"strategy": strategy, "layer": layer}
        for strategy in STRATEGIES for layer in range(1, layer_count + 1)
    ]
    routes = root["expected_routes"]
    if not isinstance(routes, list):
        raise PreflightError("expected_routes must be a list")
    for index, route in enumerate(routes):
        _exact(route, {"strategy", "layer"}, f"expected_routes[{index}]")
        if route["strategy"] not in STRATEGIES or type(route["layer"]) is not int or route["layer"] <= 0:
            raise PreflightError(f"expected_routes[{index}] has invalid strategy/layer")
    if routes != expected_routes:
        raise PreflightError("expected_routes is not the exact canonical seven-strategy layer matrix")
    return dict(root)


def _headers(extra: Mapping[str, str] | None = None) -> Dict[str, str]:
    result = {"User-Agent": "wisent-desired-results-preflight/2", "Accept": "application/json"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        result["Authorization"] = f"Bearer {token}"
    if extra:
        result.update(extra)
    return result


def _request(
    url: str, *, headers: Mapping[str, str] | None = None, limit: int,
    context: str = "HF read", method: str = "GET", body: bytes | None = None,
) -> Tuple[bytes, Mapping[str, str], int]:
    if method not in {"GET", "POST"} or (method == "GET" and body is not None):
        raise PreflightError(f"{context}: invalid read-only HTTP request")
    last: BaseException | None = None
    for attempt in range(RETRIES):
        request = urllib.request.Request(url, data=body, headers=_headers(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise PreflightError(f"{context}: response exceeds {limit} bytes: {url}")
                return data, response.headers, response.status
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise PreflightError(f"{context}: HTTP {exc.code}: {url}") from exc
        except urllib.error.URLError as exc:
            last = exc
        if attempt + 1 < RETRIES:
            time.sleep(0.25 * (2 ** attempt) + random.random() * 0.05)
    raise PreflightError(f"{context}: failed after {RETRIES} attempts: {url}: {last}") from last


def _repo_base(repo_type: str, repo_id: str) -> str:
    prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[repo_type]
    return f"https://huggingface.co/{prefix}{urllib.parse.quote(repo_id, safe='/')}"


def _resolve_url(repo_type: str, repo_id: str, revision: str, repo_path: str) -> str:
    return f"{_repo_base(repo_type, repo_id)}/resolve/{revision}/{urllib.parse.quote(repo_path, safe='/')}"


def _paths_info(
    repo_type: str, repo_id: str, revision: str, paths: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not paths or len(paths) != len(set(paths)):
        raise PreflightError("HF paths-info request paths must be nonempty and unique")
    endpoint = (
        f"https://huggingface.co/api/{repo_type}s/{urllib.parse.quote(repo_id, safe='/')}"
        f"/paths-info/{urllib.parse.quote(revision, safe='')}"
    )
    entries: Dict[str, Dict[str, Any]] = {}
    for offset in range(0, len(paths), 50):
        chunk = list(paths[offset:offset + 50])
        body = urllib.parse.urlencode({"paths": chunk, "expand": "true"}, doseq=True).encode("ascii")
        raw, _, status = _request(
            endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            limit=MAX_JSON_BYTES, context=f"HF paths-info batch {offset // 50 + 1}",
            method="POST", body=body,
        )
        if status != 200:
            raise PreflightError(f"HF paths-info batch {offset // 50 + 1}: HTTP {status}")
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PreflightError(f"HF paths-info batch {offset // 50 + 1}: malformed JSON") from exc
        if not isinstance(values, list):
            raise PreflightError(f"HF paths-info batch {offset // 50 + 1}: response is not a list")
        chunk_entries: Dict[str, Dict[str, Any]] = {}
        for index, value in enumerate(values):
            if not isinstance(value, dict) or value.get("type") != "file" or not isinstance(value.get("path"), str):
                raise PreflightError(f"HF paths-info batch {offset // 50 + 1}: invalid entry {index}")
            path = value["path"]
            if path in chunk_entries or path in entries:
                raise PreflightError(f"HF paths-info returned duplicate path: {path}")
            chunk_entries[path] = value
        if set(chunk_entries) != set(chunk):
            missing = sorted(set(chunk) - set(chunk_entries))
            extras = sorted(set(chunk_entries) - set(chunk))
            raise PreflightError(f"HF paths-info path-set mismatch; missing={missing}, extras={extras}")
        entries.update(chunk_entries)
    if set(entries) != set(paths):
        raise PreflightError("HF paths-info aggregate path-set mismatch")
    return entries


def _lfs_identity(entry: Mapping[str, Any], repo_path: str) -> Tuple[str, int]:
    if entry.get("type") != "file" or entry.get("path") != repo_path:
        raise PreflightError(f"tree metadata does not identify required file: {repo_path}")
    lfs = entry.get("lfs")
    if not isinstance(lfs, Mapping):
        raise PreflightError(f"required artifact is not immutable LFS content: {repo_path}")
    oid = _sha(lfs.get("oid"), f"tree[{repo_path}].lfs.oid")
    size = lfs.get("size")
    if type(size) is not int or size <= 0 or entry.get("size") != size:
        raise PreflightError(f"LFS size disagrees with expanded tree metadata: {repo_path}")
    return oid, size


def _pair_blob_identity(entry: Mapping[str, Any], repo_path: str) -> Dict[str, Any]:
    if entry.get("type") != "file" or entry.get("path") != repo_path:
        raise PreflightError(f"tree metadata does not identify required file: {repo_path}")
    size = entry.get("size")
    if type(size) is not int or size <= 0:
        raise PreflightError(f"tree metadata has invalid file size: {repo_path}")
    lfs = entry.get("lfs")
    if isinstance(lfs, Mapping):
        oid = _sha(lfs.get("oid"), f"tree[{repo_path}].lfs.oid")
        if lfs.get("size") != size:
            raise PreflightError(f"LFS size disagrees with expanded tree metadata: {repo_path}")
        return {"storage": "lfs", "lfs_sha256": oid, "size": size}
    return {"storage": "git", "git_oid": _sha(entry.get("oid"), f"tree[{repo_path}].oid", 40), "size": size}


def _download_json(
    repo_type: str, repo_id: str, revision: str, repo_path: str,
    identity: Mapping[str, Any],
) -> Tuple[Any, str]:
    size = identity["size"]
    if size > MAX_JSON_BYTES:
        raise PreflightError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes: {repo_path}")
    raw, _, status = _request(
        _resolve_url(repo_type, repo_id, revision, repo_path), limit=size,
        context=f"HF pair-text download {repo_path}",
    )
    content_sha256 = hashlib.sha256(raw).hexdigest()
    valid_identity = content_sha256 == identity.get("lfs_sha256") if identity["storage"] == "lfs" else (
        hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest() == identity.get("git_oid")
    )
    if status != 200 or len(raw) != size or not valid_identity:
        raise PreflightError(f"downloaded JSON does not match pinned {identity['storage']} identity: {repo_path}")
    try:
        return json.loads(raw), content_sha256
    except json.JSONDecodeError as exc:
        raise PreflightError(f"HF JSON artifact is malformed: {repo_path}") from exc


def _range(repo_type: str, repo_id: str, revision: str, repo_path: str, start: int, end: int) -> bytes:
    raw, headers, status = _request(
        _resolve_url(repo_type, repo_id, revision, repo_path),
        headers={"Range": f"bytes={start}-{end}", "Accept": "application/octet-stream"},
        limit=end - start + 1,
        context=f"HF range {repo_path} bytes {start}-{end}",
    )
    content_range = headers.get("Content-Range", "")
    if status != 206 or content_range.split("/", 1)[0] != f"bytes {start}-{end}" or len(raw) != end - start + 1:
        raise PreflightError(f"HF did not honor exact byte range {start}-{end}: {repo_path}")
    return raw


def _canonical_pairs(
    artifact: Any, expected_pairs: int,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]], str, str]:
    if not isinstance(artifact, Mapping):
        raise PreflightError("pair-text artifact must be an object")
    if "pairs" in artifact:
        if not isinstance(artifact["pairs"], list) or any(
            isinstance(key, str) and key.isascii() and key.isdigit() for key in artifact if key != "pairs"
        ):
            raise PreflightError("pair-text artifact mixes or malforms schema variants")
        variant = "standard_pairs_v1"
        raw_pairs = artifact["pairs"]
        normalized: list[Dict[str, Any]] = []
        for pair_id, value in enumerate(raw_pairs):
            if not isinstance(value, Mapping):
                raise PreflightError(f"pair text {pair_id} is not an object")
            if "pair_id" in value and (type(value["pair_id"]) is not int or value["pair_id"] != pair_id):
                raise PreflightError(f"pair text {pair_id} has a conflicting pair_id")
            if not isinstance(value.get("prompt"), str):
                raise PreflightError(f"pair text {pair_id} lacks string prompt")
            responses: Dict[str, str] = {}
            for field in ("positive_response", "negative_response"):
                response = value.get(field)
                if not isinstance(response, Mapping) or not isinstance(response.get("model_response"), str):
                    raise PreflightError(f"pair text {pair_id} has invalid {field}")
                responses[field] = response["model_response"]
            canonical = {
                key: item for key, item in value.items()
                if key not in {"pair_id", "prompt", "positive_response", "negative_response"}
            }
            canonical.update({
                "pair_id": pair_id, "prompt": value["prompt"],
                "positive_response": {"model_response": responses["positive_response"]},
                "negative_response": {"model_response": responses["negative_response"]},
            })
            normalized.append(canonical)
    else:
        keys = list(artifact)
        if any(
            not isinstance(key, str) or not key.isascii() or not key.isdigit()
            or (key != "0" and key.startswith("0"))
            for key in keys
        ):
            raise PreflightError("numeric-key pair schema contains a noncanonical key")
        expected_keys = [str(pair_id) for pair_id in range(len(keys))]
        if set(keys) != set(expected_keys):
            raise PreflightError("numeric-key pair schema must be exactly contiguous 0..N-1")
        required_text = {"prompt", "positive", "negative"}
        required_metadata = required_text | {"metadata"}
        row_key_sets = [set(artifact[key]) if isinstance(artifact[key], Mapping) else None for key in expected_keys]
        if row_key_sets and all(fields == required_metadata for fields in row_key_sets):
            variant = "numeric_key_pairs_v1"
            include_metadata = True
        elif row_key_sets and all(fields == required_text for fields in row_key_sets):
            variant = "numeric_key_direct_response_no_metadata"
            include_metadata = False
        else:
            raise PreflightError(
                "numeric-key rows must uniformly have exactly prompt/positive/negative, with or without metadata"
            )
        normalized = []
        for pair_id, key in enumerate(expected_keys):
            value = artifact[key]
            if any(not isinstance(value[field], str) for field in ("prompt", "positive", "negative")):
                raise PreflightError(f"numeric-key pair {pair_id} has non-string text")
            normalized.append({
                "pair_id": pair_id, "prompt": value["prompt"],
                "positive_response": {"model_response": value["positive"]},
                "negative_response": {"model_response": value["negative"]},
                "metadata": value["metadata"] if include_metadata else {},
            })
    if len(normalized) < expected_pairs:
        raise PreflightError("pair-text artifact does not cover expected_pairs")
    selected: list[Dict[str, Any]] = []
    support: list[Dict[str, Any]] = []
    stable_seen: set[str] = set()
    for pair_id, canonical_value in enumerate(normalized[:expected_pairs]):
        canonical = dict(canonical_value)
        identity_value = {key: item for key, item in canonical.items() if key != "stable_id"}
        derived_stable = hashlib.sha256(canonical_json(identity_value)).hexdigest()
        stable_id = canonical.get("stable_id", derived_stable)
        if not isinstance(stable_id, str) or not stable_id or stable_id in stable_seen:
            raise PreflightError(f"pair text {pair_id} has an invalid or duplicate stable_id")
        stable_seen.add(stable_id)
        canonical["stable_id"] = stable_id
        selected.append(canonical)
        support.append({"pair_id": pair_id, "stable_id": stable_id})
    return selected, support, canonical_sha256(selected), variant


def _split_support(rows: list[Dict[str, Any]], target_id: str) -> Dict[str, list[Dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        canonical_json({"target_id": target_id, **row})
    ).digest())
    train_end = len(rows) * 60 // 100
    validation_end = train_end + len(rows) * 20 // 100
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def _route_proof(descriptor: Mapping[str, Any], route: Mapping[str, Any], entry: Mapping[str, Any]) -> Dict[str, Any]:
    target, source = descriptor["target"], descriptor["source_evidence"]
    strategy, layer = route["strategy"], route["layer"]
    repo_path = f"activations/{target['model_slug']}/{target['benchmark']}/{strategy}/layer_{layer}.safetensors"
    oid, size = _lfs_identity(entry, repo_path)
    if size < 10:
        raise PreflightError(f"safetensors artifact is too small: {repo_path}")
    probe_end = min(size - 1, HEADER_PROBE_BYTES - 1)
    probe = _range(
        source["activation_repo_type"], source["activation_repo_id"],
        source["activation_revision"], repo_path, 0, probe_end,
    )
    header_length = struct.unpack("<Q", probe[:8])[0]
    header_end = 8 + header_length
    if header_length <= 1 or header_length > MAX_HEADER_BYTES or header_end > size:
        raise PreflightError(f"invalid safetensors header length for {repo_path}")
    raw_header = probe[8:header_end]
    if header_end > len(probe):
        raw_header += _range(
            source["activation_repo_type"], source["activation_repo_id"],
            source["activation_revision"], repo_path, len(probe), header_end - 1,
        )
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid safetensors header JSON: {repo_path}") from exc
    if not isinstance(header, Mapping):
        raise PreflightError(f"safetensors header is not an object: {repo_path}")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("pair_ids"), str):
        raise PreflightError(f"safetensors metadata lacks pair_ids: {repo_path}")
    try:
        pair_ids = json.loads(metadata["pair_ids"])
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid pair_ids metadata: {repo_path}") from exc
    expected_ids = list(range(target["expected_pairs"]))
    if pair_ids != expected_ids:
        raise PreflightError(f"activation pair_ids do not exactly match canonical support: {repo_path}")
    shapes: Dict[str, list[int]] = {}
    dtypes: Dict[str, str] = {}
    data_size = size - 8 - header_length
    dtype_sizes = {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
        "U32": 4, "I32": 4, "F32": 4,
        "U64": 8, "I64": 8, "F64": 8,
    }
    intervals: list[Tuple[int, int]] = []
    for name in ("pos_activations", "neg_activations"):
        tensor = header.get(name)
        if not isinstance(tensor, Mapping) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise PreflightError(f"invalid {name} tensor descriptor: {repo_path}")
        shape, dtype, offsets = tensor["shape"], tensor["dtype"], tensor["data_offsets"]
        if (not isinstance(shape, list) or len(shape) != 2 or shape[0] != target["expected_pairs"]
                or type(shape[1]) is not int or shape[1] <= 0 or dtype not in dtype_sizes
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or offsets[0] < 0 or offsets[1] <= offsets[0] or offsets[1] > data_size):
            raise PreflightError(f"invalid {name} shape/dtype/offsets: {repo_path}")
        expected_bytes = shape[0] * shape[1] * dtype_sizes[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise PreflightError(f"{name} byte span does not match shape/dtype: {repo_path}")
        intervals.append((offsets[0], offsets[1]))
        shapes[name], dtypes[name] = shape, dtype
    if shapes["pos_activations"] != shapes["neg_activations"] or dtypes["pos_activations"] != dtypes["neg_activations"]:
        raise PreflightError(f"positive/negative tensor contracts differ: {repo_path}")
    if intervals[0][1] > intervals[1][0] and intervals[1][1] > intervals[0][0]:
        raise PreflightError(f"positive/negative tensor byte spans overlap: {repo_path}")
    return {
        "schema_version": 2,
        "proof_kind": "pinned_hf_safetensors_header",
        "target_id": target["target_id"],
        "activation_artifact": {
            "repo_id": source["activation_repo_id"], "repo_type": source["activation_repo_type"],
            "revision": source["activation_revision"], "path": repo_path,
            "lfs_sha256": oid, "size": size,
        },
        "route": {"strategy": strategy, "layer": layer},
        "pair_ids": pair_ids,
        "tensor_shapes": shapes,
        "tensor_dtypes": dtypes,
        "safetensors_header_length": header_length,
        "safetensors_header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "tensor_payload_downloaded": False,
    }


def _encoded(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"


def _write_content_addressed(directory: Path, stem: str, value: Any) -> Tuple[Path, Dict[str, str]]:
    encoded = _encoded(value)
    digest = hashlib.sha256(encoded).hexdigest()
    path = directory / f"{stem}.{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path, {"sha256": digest, "size": str(len(encoded)), "generation": f"sha256:{digest}"}


def _bundle_ref(path: Path, staging: Path, target: Mapping[str, Any], identity: Mapping[str, str]) -> Dict[str, str]:
    relative = path.relative_to(staging).as_posix()
    return {
        "uri": f"bundle:///{relative}",
        "generation": identity["generation"], "size": identity["size"], "sha256": identity["sha256"],
    }


def run(descriptor_path: Path, output: Path, workers: int) -> Path:
    descriptor_path, output = descriptor_path.resolve(), output.resolve()
    descriptor = _read_descriptor(descriptor_path)
    target, source = descriptor["target"], descriptor["source_evidence"]
    if output.exists():
        raise PreflightError(f"destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        pair_path = f"pair_texts/{target['benchmark']}.json"
        route_paths = {
            (route["strategy"], route["layer"]):
                f"activations/{target['model_slug']}/{target['benchmark']}/{route['strategy']}/layer_{route['layer']}.safetensors"
            for route in descriptor["expected_routes"]
        }
        requested_paths = [pair_path, *route_paths.values()]
        path_entries = _paths_info(
            source["activation_repo_type"], source["activation_repo_id"],
            source["activation_revision"], requested_paths,
        )
        pair_identity = _pair_blob_identity(path_entries[pair_path], pair_path)
        pair_artifact, pair_content_sha256 = _download_json(
            source["activation_repo_type"], source["activation_repo_id"],
            source["activation_revision"], pair_path, pair_identity,
        )
        selected, support_rows, pair_text_hash, pair_schema_variant = _canonical_pairs(
            pair_artifact, target["expected_pairs"]
        )
        try:
            pair_source = validate_pair_source({
                "repo_id": source["activation_repo_id"],
                "repo_type": source["activation_repo_type"],
                "revision": source["activation_revision"],
                "path": pair_path,
                **pair_identity,
                "content_sha256": pair_content_sha256,
                "schema_variant": pair_schema_variant,
            }, "pair source")
        except PairSourceError as exc:
            raise PreflightError(str(exc)) from exc
        splits = _split_support(support_rows, target["target_id"])
        support_payload = {
            "schema_version": 2, "proof_kind": "deterministic_support_split",
            "target_id": target["target_id"], "pair_count": target["expected_pairs"],
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "pair_text_hash": pair_text_hash,
            "pair_text_source": pair_source,
            "split_algorithm": "sha256(canonical_json({target_id,pair_id,stable_id}));60/20/20",
            "split_counts": {name: len(rows) for name, rows in splits.items()}, "splits": splits,
            "support_sha256": canonical_sha256(splits),
        }
        pair_payload = {
            "schema_version": 2, "target_id": target["target_id"],
            "source": pair_source,
            "pair_count": len(selected), "pair_text_hash": pair_text_hash, "pairs": selected,
        }
        target_root = staging.joinpath(target["model_slug"], *target["benchmark"].split("/"))
        try:
            target_root.relative_to(staging)
        except ValueError as exc:
            raise PreflightError("target bundle root escapes atomic staging directory") from exc
        support_file, support_identity = _write_content_addressed(target_root, "support-proof", support_payload)
        pair_file, pair_identity = _write_content_addressed(target_root, "pair-texts-selected", pair_payload)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_routes = {
                executor.submit(
                    _route_proof, descriptor, route,
                    path_entries[route_paths[(route["strategy"], route["layer"])]],
                ): route
                for route in descriptor["expected_routes"]
            }
            proofs: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for future in concurrent.futures.as_completed(future_routes):
                route = future_routes[future]
                proofs[(route["strategy"], route["layer"])] = future.result()
        widths = {proof["tensor_shapes"]["pos_activations"][1] for proof in proofs.values()}
        dtypes = {proof["tensor_dtypes"]["pos_activations"] for proof in proofs.values()}
        if len(proofs) != len(route_paths) or len(widths) != 1 or len(dtypes) != 1:
            raise PreflightError("route matrix is incomplete or has inconsistent activation shape/dtype")
        manifest_routes = []
        index_routes = []
        for route in descriptor["expected_routes"]:
            strategy, layer = route["strategy"], route["layer"]
            proof = proofs[(strategy, layer)]
            proof_file, proof_identity = _write_content_addressed(target_root / "route-proofs", f"{strategy}.layer_{layer}.proof", proof)
            proof_ref = _bundle_ref(proof_file, staging, target, proof_identity)
            completion = {
                "schema_version": 2, "complete": True, "target_id": target["target_id"],
                "route": route, "proof_ref": proof_ref,
                "activation_lfs_sha256": proof["activation_artifact"]["lfs_sha256"],
                "activation_header_sha256": proof["safetensors_header_sha256"],
            }
            completion_file, completion_identity = _write_content_addressed(target_root / "route-completions", f"{strategy}.layer_{layer}.completion", completion)
            completion_ref = _bundle_ref(completion_file, staging, target, completion_identity)
            manifest_routes.append({"strategy": strategy, "layer": layer, "completion_ref": completion_ref, "proof_ref": proof_ref})
            index_routes.append({"strategy": strategy, "layer": layer, "completion_ref": completion_ref, "proof_ref": proof_ref})
        completion_index = {
            "schema_version": 2, "complete": True, "target_id": target["target_id"],
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "activation_cache_sha256": source["activation_cache_sha256"],
            "activation_record_sha256": source["activation_record_sha256"],
            "pair_texts_ref": _bundle_ref(pair_file, staging, target, pair_identity),
            "support_proof_ref": _bundle_ref(support_file, staging, target, support_identity),
            "route_count": len(index_routes), "routes": index_routes,
            "submission_performed": False, "model_loaded": False, "tensor_payload_downloaded": False,
        }
        index_file, index_identity = _write_content_addressed(target_root, "completion-index", completion_index)
        manifest_payload = {
            "schema_version": 2,
            "protocol": descriptor["protocol"],
            "target": {
                "target_id": target["target_id"], "result_id": target["result_id"],
                "model_name": target["model_name"], "model_slug": target["model_slug"],
                "benchmark": target["benchmark"], "expected_pairs": target["expected_pairs"],
                "result_prefix": f"results/{descriptor['protocol']['id']}/{target['model_slug']}/{target['benchmark']}",
            },
            "revisions": {
                "inventory_sha256": source["inventory_sha256"], "activation_revision": source["activation_revision"],
                "model_revision": source["model_revision"], "tokenizer_revision": source["tokenizer_revision"],
            },
            "activation": {
                "status": "complete", "eligible": True, "layer_count": target["layer_count"],
                "n_pairs": target["expected_pairs"], "grouped": False,
                "strategies": {name: target["layer_count"] for name in STRATEGIES},
                "routes": manifest_routes,
                "proof": {"cache_sha256": source["activation_cache_sha256"], "record_sha256": source["activation_record_sha256"]},
            },
            "support": {
                "state": "prepared", "proof_sha256": support_identity["sha256"],
                "pair_texts_ref": _bundle_ref(pair_file, staging, target, pair_identity),
                "pair_count": target["expected_pairs"], "split_counts": support_payload["split_counts"], "splits": splits,
            },
            "evaluation": {"required_outputs": ["accuracy", "coherence"], "split": "test"},
            "calibration": {"methods": list(METHODS), "strategies": list(STRATEGIES),
                            "layer_count": target["layer_count"], "expected_pairs": target["expected_pairs"]},
            "execution": {"state": "unprepared", "blocked": False, "rerun_locked": False,
                          "publication": None, "provenance": {"execution_sha256": None, "contract_sha256": None}},
        }
        manifest = finalize_target_manifest(manifest_payload)
        manifest_file, manifest_identity = _write_content_addressed(target_root, "target-manifest", manifest)
        bundle_index = {
            "schema_version": 2, "bundle_kind": "activation_preflight",
            "target_id": target["target_id"], "descriptor_sha256": descriptor["descriptor_sha256"],
            "completion_index_ref": _bundle_ref(index_file, staging, target, index_identity),
            "target_manifest_ref": _bundle_ref(manifest_file, staging, target, manifest_identity),
        }
        bundle_file, _ = _write_content_addressed(staging, "bundle-index", bundle_index)
        os.replace(staging, output)
        return output / bundle_file.name
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise



def _remote_artifact_ref(value: Any, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"uri", "generation", "sha256", "size"}:
        raise PreflightError(f"{label} must be an exact ArtifactRef")
    result = {key: value[key] for key in ("uri", "generation", "sha256", "size")}
    if (not all(isinstance(result[key], str) for key in result)
            or not result["uri"].startswith("gs://")
            or not result["generation"].isdigit()):
        raise PreflightError(f"{label} must identify an exact GCS generation")
    _sha(result["sha256"], f"{label}.sha256")
    if (not result["size"].isdigit() or result["size"].startswith("0")):
        raise PreflightError(f"{label}.size must be a canonical positive decimal byte length")
    return result


def _store_read_exact(store: Any, ref: Mapping[str, Any], label: str) -> bytes:
    exact = _remote_artifact_ref(ref, label)
    try:
        result = store.read(exact["uri"], exact["generation"])
    except TypeError:
        result = store.read(exact)
    if isinstance(result, tuple):
        if len(result) != 2 or not isinstance(result[0], bytes) or str(result[1]) != exact["generation"]:
            raise PreflightError(f"{label} store read returned a different generation")
        data = result[0]
    elif isinstance(result, bytes):
        data = result
    else:
        raise PreflightError(f"{label} store read did not return bytes")
    if len(data) != int(exact["size"]) or hashlib.sha256(data).hexdigest() != exact["sha256"]:
        raise PreflightError(f"{label} bytes differ from their exact ArtifactRef")
    return data


def _decode_canonical_remote(store: Any, ref: Mapping[str, Any], label: str,
                             *, newline: bool = False) -> Tuple[Dict[str, Any], Dict[str, str]]:
    exact = _remote_artifact_ref(ref, f"{label}_ref")
    raw = _store_read_exact(store, exact, f"{label}_ref")
    payload = raw[:-1] if newline and raw.endswith(b"\n") else raw
    if not payload or payload.endswith(b"\n"):
        raise PreflightError(f"{label} has invalid trailing bytes")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{label} is not strict ASCII JSON") from exc
    if canonical_json(value) != payload or not isinstance(value, Mapping):
        raise PreflightError(f"{label} is not a canonical JSON object")
    return dict(value), exact


def _read_remote_inventory_plan(store: Any, ref: Mapping[str, Any]) -> Dict[str, Any]:
    plan, _ = _decode_canonical_remote(store, ref, "inventory plan", newline=True)
    if plan.get("schema_version") not in {2, 3}:
        raise PreflightError("inventory plan schema_version must be 2 or 3")
    plan_sha256 = _sha(plan.get("plan_sha256"), "inventory plan.plan_sha256")
    unhashed = dict(plan)
    del unhashed["plan_sha256"]
    if canonical_sha256(unhashed) != plan_sha256:
        raise PreflightError("inventory plan.plan_sha256 does not match its logical payload")
    _sha(plan.get("inventory_sha256"), "inventory plan.inventory_sha256")
    if any(field in plan for field in (
            "selection_ref", "selection_sha256", "submission_ref", "submission_sha256")):
        raise PreflightError("inventory plan must not contain circular submission or selection lineage")
    if plan.get("no_submit") is not True:
        raise PreflightError("inventory plan must be explicitly non-submitting")
    if plan.get("descriptor_kind") != "activation_proof_preflight":
        raise PreflightError("inventory plan descriptor_kind is invalid")
    descriptors = plan.get("descriptors")
    if not isinstance(descriptors, list):
        raise PreflightError("inventory plan.descriptors must be a list")
    if plan.get("descriptor_count", len(descriptors)) != len(descriptors):
        raise PreflightError("inventory plan.descriptor_count differs from descriptors")
    target_ids: set[str] = set()
    for index, entry in enumerate(descriptors):
        if not isinstance(entry, Mapping):
            raise PreflightError(f"inventory plan.descriptors[{index}] must be an object")
        target_id_value = entry.get("target_id")
        if (not isinstance(target_id_value, str) or not target_id_value
                or target_id_value in target_ids):
            raise PreflightError("inventory plan descriptor target IDs must be non-empty and unique")
        target_ids.add(target_id_value)
        descriptor_sha = _sha(
            entry.get("descriptor_sha256"),
            f"inventory descriptor {target_id_value}.descriptor_sha256",
        )
        descriptor_ref = _remote_artifact_ref(
            entry.get("descriptor_ref"),
            f"inventory descriptor {target_id_value}.descriptor_ref",
        )
        if not descriptor_ref["uri"].endswith(f"/descriptors/{descriptor_sha}.json"):
            raise PreflightError(f"inventory descriptor {target_id_value} ref is not at its canonical path")
    return plan


def _read_remote_submission(store: Any, ref: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    submission, exact = _decode_canonical_remote(store, ref, "submission manifest")
    _exact(submission, {
        "schema_version", "plan_kind", "inventory_plan_ref", "inventory_plan_sha256",
        "selection_ref", "selection_sha256", "target_count", "targets", "plan_sha256",
    }, "submission manifest")
    if (submission["schema_version"] != 3
            or submission["plan_kind"] != "desired-results-preflight-plan-v3"):
        raise PreflightError("submission manifest schema/version is invalid")
    plan_sha = _sha(submission["plan_sha256"], "submission manifest.plan_sha256")
    unhashed = dict(submission)
    del unhashed["plan_sha256"]
    if canonical_sha256(unhashed) != plan_sha:
        raise PreflightError("submission manifest.plan_sha256 does not match its logical payload")
    if not exact["uri"].endswith(f"/preflight/submissions/{plan_sha}.json"):
        raise PreflightError("submission_ref is not at its canonical content-addressed path")
    _remote_artifact_ref(submission["inventory_plan_ref"], "submission inventory_plan_ref")
    _sha(submission["inventory_plan_sha256"], "submission inventory_plan_sha256")
    selection_ref = submission["selection_ref"]
    selection_sha = submission["selection_sha256"]
    if selection_ref is None or selection_sha is None:
        if selection_ref is not None or selection_sha is not None:
            raise PreflightError("submission selection_ref and selection_sha256 must both be null or present")
    else:
        _remote_artifact_ref(selection_ref, "submission selection_ref")
        _sha(selection_sha, "submission selection_sha256")
    targets = submission["targets"]
    if not isinstance(targets, list) or not targets or submission["target_count"] != len(targets):
        raise PreflightError("submission target_count differs from its non-empty targets")
    target_ids: list[str] = []
    node_ids: set[str] = set()
    for index, value in enumerate(targets):
        target = _exact(value, {
            "target_id", "descriptor_sha256", "descriptor_ref", "output_prefix", "node_id",
        }, f"submission targets[{index}]")
        target_id_value = _string(target["target_id"], f"submission targets[{index}].target_id")
        descriptor_sha = _sha(
            target["descriptor_sha256"], f"submission target {target_id_value}.descriptor_sha256",
        )
        descriptor_ref = _remote_artifact_ref(
            target["descriptor_ref"], f"submission target {target_id_value}.descriptor_ref",
        )
        if not descriptor_ref["uri"].endswith(f"/descriptors/{descriptor_sha}.json"):
            raise PreflightError(f"submission target {target_id_value} descriptor ref is not canonical")
        target_key = hashlib.sha256(target_id_value.encode("utf-8")).hexdigest()[:16]
        suffix = f"/preflight/{target_key}/{descriptor_sha}"
        output_prefix = target["output_prefix"]
        if (not isinstance(output_prefix, str) or not output_prefix.startswith("gs://")
                or not output_prefix.rstrip("/").endswith(suffix)):
            raise PreflightError(f"submission target {target_id_value} output_prefix is not canonical")
        node_payload = {
            "target_id": target_id_value,
            "descriptor_sha256": descriptor_sha,
            "descriptor_ref": descriptor_ref,
            "output_prefix": output_prefix,
        }
        expected_node_id = f"preflight-node-v3:{canonical_sha256(node_payload)}"
        if target["node_id"] != expected_node_id or target["node_id"] in node_ids:
            raise PreflightError(f"submission target {target_id_value} node_id is invalid or duplicated")
        target_ids.append(target_id_value)
        node_ids.add(target["node_id"])
    if target_ids != sorted(target_ids) or len(set(target_ids)) != len(target_ids):
        raise PreflightError("submission targets must have unique target_ids sorted canonically")
    return submission, exact


def _read_remote_selection(store: Any, ref: Mapping[str, Any], inventory_sha256: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    selection, exact = _decode_canonical_remote(store, ref, "inventory selection")
    try:
        validate_inventory_selection(selection, inventory_sha256)
    except SelectionError as exc:
        raise PreflightError(str(exc)) from exc
    return selection, exact


def _bind_remote_descriptor(plan: Mapping[str, Any], descriptor: Mapping[str, Any],
                            descriptor_ref: Mapping[str, str]) -> None:
    target = descriptor["target"]
    evidence = descriptor["source_evidence"]
    if evidence["inventory_sha256"] != plan["inventory_sha256"]:
        raise PreflightError("descriptor source evidence binds a different inventory")
    matching = [
        entry for entry in plan["descriptors"]
        if entry["target_id"] == target["target_id"]
    ]
    if len(matching) != 1:
        raise PreflightError("descriptor target is not uniquely selected by the inventory plan")
    binding = matching[0]
    if binding["descriptor_sha256"] != descriptor["descriptor_sha256"]:
        raise PreflightError("inventory plan binds a different descriptor logical hash")
    if _remote_artifact_ref(binding["descriptor_ref"], "inventory descriptor_ref") != descriptor_ref:
        raise PreflightError("inventory plan binds a different descriptor ArtifactRef")


def _create_identical(store: Any, uri: str, data: bytes) -> Dict[str, str]:
    try:
        created = store.create(uri, data)
        if isinstance(created, Mapping):
            ref = created
        else:
            ref = {
                "uri": uri, "generation": str(created), "size": str(len(data)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    except Exception as create_error:
        resolve = getattr(store, "resolve", None)
        ref = None if resolve is None else resolve(uri)
        if ref is None:
            raise PreflightError(f"create-only upload failed for {uri}: {create_error}") from create_error
        try:
            existing = _store_read_exact(store, ref, f"existing immutable output {uri}")
        except Exception as read_error:
            raise PreflightError(f"cannot verify existing immutable output {uri}: {read_error}") from create_error
        if existing != data:
            raise PreflightError(f"immutable output conflict at {uri}") from create_error
    exact = _remote_artifact_ref(ref, f"published output {uri}")
    if exact["uri"] != uri or exact["sha256"] != hashlib.sha256(data).hexdigest() or exact["size"] != str(len(data)):
        raise PreflightError(f"store returned a ref that does not identify uploaded bytes: {uri}")
    return exact


def execute_remote(descriptor_ref: Mapping[str, Any], output_prefix: str,
                   terminal_receipt_uri: str, *, node_id: str,
                   submission_ref: Mapping[str, Any], workers: int, store: Any,
                   run_function: Any = run) -> Dict[str, Any]:
    """Exact-load one submitted node and all lineage before any descriptor/HF work."""
    descriptor_exact = _remote_artifact_ref(descriptor_ref, "descriptor_ref")
    submission, submitted_exact = _read_remote_submission(store, submission_ref)
    plan_exact = _remote_artifact_ref(
        submission["inventory_plan_ref"], "submission inventory_plan_ref",
    )
    plan = _read_remote_inventory_plan(store, plan_exact)
    plan_sha256 = plan["plan_sha256"]
    if submission["inventory_plan_sha256"] != plan_sha256:
        raise PreflightError("submission inventory_plan_sha256 differs from the loaded inventory plan")
    if not plan_exact["uri"].endswith(f"/inventory-plans/{plan_sha256}.json"):
        raise PreflightError("inventory_plan_ref is not at its canonical content-addressed path")

    selected_exact: Dict[str, str] | None = None
    selected_sha256: str | None = None
    selected_target_ids: set[str] | None = None
    if submission["selection_ref"] is not None:
        selection, selected_exact = _read_remote_selection(
            store, submission["selection_ref"], plan["inventory_sha256"],
        )
        selected_sha256 = selection.get("selection_sha256", selection.get("content_sha256"))
        if submission["selection_sha256"] != selected_sha256:
            raise PreflightError("submission selection_sha256 differs from the loaded selection seal")
        selected_target_ids = set(selection["target_ids"])

    inventory_by_target = {entry["target_id"]: entry for entry in plan["descriptors"]}
    submission_by_target = {target["target_id"]: target for target in submission["targets"]}
    if selected_target_ids is not None and not selected_target_ids <= set(inventory_by_target):
        raise PreflightError("loaded selection contains a target absent from the inventory plan")
    expected_target_ids = set(inventory_by_target) if selected_target_ids is None else selected_target_ids
    if set(submission_by_target) != expected_target_ids:
        raise PreflightError("submission target membership differs from its loaded inventory selection")
    for submitted_target_id, submitted_target in submission_by_target.items():
        inventory_binding = inventory_by_target[submitted_target_id]
        if (inventory_binding["descriptor_sha256"] != submitted_target["descriptor_sha256"]
                or _remote_artifact_ref(
                    inventory_binding["descriptor_ref"],
                    f"inventory descriptor {submitted_target_id}.descriptor_ref",
                ) != _remote_artifact_ref(
                    submitted_target["descriptor_ref"],
                    f"submission target {submitted_target_id}.descriptor_ref",
                )):
            raise PreflightError(
                f"submission target {submitted_target_id} descriptor binding differs from the loaded inventory"
            )

    matching_nodes = [target for target in submission["targets"] if target["node_id"] == node_id]
    if len(matching_nodes) != 1:
        raise PreflightError("node_id is not uniquely authorized by the submission manifest")
    submitted_node = matching_nodes[0]
    target_id_value = submitted_node["target_id"]
    if descriptor_exact != _remote_artifact_ref(
            submitted_node["descriptor_ref"], "submission node descriptor_ref"):
        raise PreflightError("descriptor_ref differs from the submitted node binding")
    if output_prefix != submitted_node["output_prefix"]:
        raise PreflightError("output_prefix differs from the submitted node path")
    expected_terminal_uri = output_prefix.rstrip("/") + "/bundle-completion/exact.json"
    if terminal_receipt_uri != expected_terminal_uri:
        raise PreflightError("terminal receipt URI is not the exact submitted node terminal path")

    descriptor_bytes = _store_read_exact(store, descriptor_exact, "descriptor_ref")
    try:
        descriptor_value = json.loads(descriptor_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("downloaded descriptor is not strict JSON") from exc
    supplied_descriptor_sha = descriptor_value.get("descriptor_sha256") if isinstance(descriptor_value, Mapping) else None
    if supplied_descriptor_sha is None:
        raise PreflightError("downloaded descriptor lacks descriptor_sha256")
    if descriptor_bytes != canonical_json(descriptor_value) + b"\n":
        raise PreflightError("downloaded descriptor bytes are not canonical newline-terminated JSON")
    temporary_root = Path(tempfile.mkdtemp(prefix="desired-results-preflight-worker."))
    try:
        local_descriptor = temporary_root / "descriptor.json"
        local_descriptor.write_bytes(descriptor_bytes)
        # All immutable local and remote bindings fail before the first HF request.
        descriptor = _read_descriptor(local_descriptor)
        _bind_remote_descriptor(plan, descriptor, descriptor_exact)
        local_bundle = temporary_root / "bundle"
        bundle_index_path = Path(run_function(local_descriptor, local_bundle, workers)).resolve()
        try:
            bundle_index_path.relative_to(local_bundle.resolve())
        except ValueError as exc:
            raise PreflightError("local preflight returned a bundle index outside its output directory") from exc
        if not bundle_index_path.is_file():
            raise PreflightError("local preflight returned a missing bundle index")
        published: Dict[str, Tuple[Dict[str, str], Dict[str, Any]]] = {}
        active: set[str] = set()
        raw_sha_to_relative = {
            hashlib.sha256(candidate.read_bytes()).hexdigest(): candidate.relative_to(local_bundle).as_posix()
            for candidate in local_bundle.rglob("*") if candidate.is_file()
        }

        def publish_relative(relative: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
            if relative in published:
                return published[relative]
            candidate = (local_bundle / relative).resolve()
            try:
                candidate.relative_to(local_bundle.resolve())
            except ValueError as exc:
                raise PreflightError("bundle ref escapes local output directory") from exc
            if relative in active or not candidate.is_file():
                raise PreflightError(f"bundle contains a cyclic or missing ref: {relative}")
            active.add(relative)
            try:
                value = json.loads(candidate.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PreflightError(f"bundle artifact is not strict JSON: {relative}") from exc

            def rewrite(item: Any) -> Any:
                if isinstance(item, Mapping):
                    if set(item) == {"uri", "generation", "sha256", "size"} and isinstance(item.get("uri"), str) and item["uri"].startswith("bundle:///"):
                        dependency = item["uri"][10:]
                        return dict(publish_relative(dependency)[0])
                    rewritten_mapping = {key: rewrite(child) for key, child in item.items()}
                    support_digest = rewritten_mapping.get("proof_sha256")
                    if isinstance(support_digest, str) and support_digest in raw_sha_to_relative:
                        rewritten_mapping["proof_sha256"] = publish_relative(raw_sha_to_relative[support_digest])[0]["sha256"]
                    if "manifest_sha256" in rewritten_mapping:
                        unhashed_manifest = dict(rewritten_mapping)
                        del unhashed_manifest["manifest_sha256"]
                        rewritten_mapping["manifest_sha256"] = canonical_sha256(unhashed_manifest)
                    return rewritten_mapping
                if isinstance(item, list):
                    return [rewrite(child) for child in item]
                return item

            rewritten = rewrite(value)
            data = canonical_json(rewritten)
            if b"bundle:///" in data or b"file://" in data:
                raise PreflightError(f"local reference survived production rebinding: {relative}")
            digest = hashlib.sha256(data).hexdigest()
            uri = output_prefix.rstrip("/") + "/artifacts/" + digest + ".json"
            ref = _create_identical(store, uri, data)
            active.remove(relative)
            published[relative] = (ref, rewritten)
            return ref, rewritten

        bundle_relative = bundle_index_path.relative_to(local_bundle.resolve()).as_posix()
        bundle_ref, bundle = publish_relative(bundle_relative)
        # Upload even unreferenced bundle files; a successful receipt certifies the complete local output.
        for candidate in sorted(path for path in local_bundle.rglob("*") if path.is_file()):
            publish_relative(candidate.relative_to(local_bundle).as_posix())
        if (bundle.get("descriptor_sha256") != supplied_descriptor_sha
                or bundle.get("target_id") != target_id_value):
            raise PreflightError("bundle index is not bound to the submitted target descriptor")
        completion_ref = _remote_artifact_ref(bundle.get("completion_index_ref"), "bundle completion ref")
        manifest_ref = _remote_artifact_ref(bundle.get("target_manifest_ref"), "bundle manifest ref")
        by_uri = {ref["uri"]: value for ref, value in published.values()}
        completion = by_uri.get(completion_ref["uri"])
        manifest = by_uri.get(manifest_ref["uri"])
        if completion is None or manifest is None:
            raise PreflightError("bundle terminal documents were not published")
        pair_texts_ref = _remote_artifact_ref(completion.get("pair_texts_ref"), "completion pair texts ref")
        support_proof_ref = _remote_artifact_ref(completion.get("support_proof_ref"), "completion support proof ref")
        receipt = {
            "node_id": node_id, "status": "complete",
            "descriptor_sha256": supplied_descriptor_sha,
            "target_id": bundle["target_id"],
            "inventory_plan_ref": plan_exact,
            "inventory_plan_sha256": plan_sha256,
            "selection_ref": selected_exact,
            "selection_sha256": selected_sha256,
            "submission_ref": submitted_exact,
            "submission_sha256": submission["plan_sha256"],
            "bundle_index_ref": bundle_ref, "bundle_index": bundle,
            "completion_index_ref": completion_ref, "completion_index": completion,
            "pair_texts_ref": pair_texts_ref, "support_proof_ref": support_proof_ref,
            "target_manifest_ref": manifest_ref, "target_manifest": manifest,
        }
        # The terminal marker is deliberately the final write.  Failures leave no partial marker.
        _create_identical(store, terminal_receipt_uri, canonical_json(receipt))
        return receipt
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.workers < 1 or args.workers > 32:
            raise PreflightError("--workers must be between 1 and 32")
        print(run(args.descriptor, args.output, args.workers))
        return 0
    except (PreflightError, OSError, ValueError) as exc:
        print(f"activation preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
