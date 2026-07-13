#!/usr/bin/env python3
"""Deterministic, leakage-safe static analysis of immutable activation bundles."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import struct
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .desired_results_execution_contract import (
        ContractError, artifact_ref, canonical_json, canonical_sha256,
        observe_runtime_evidence, runtime_evidence_sha256, validate_artifact_ref,
    )
    from .desired_results_pair_source import PairSourceError, validate_pair_source
    from .desired_results_target import validate_target_manifest
except ImportError:
    try:
        from scripts.steering.desired_results_execution_contract import (
            ContractError, artifact_ref, canonical_json, canonical_sha256,
            observe_runtime_evidence, runtime_evidence_sha256, validate_artifact_ref,
        )
        from scripts.steering.desired_results_pair_source import PairSourceError, validate_pair_source
        from scripts.steering.desired_results_target import validate_target_manifest
    except ImportError:
        from desired_results_execution_contract import (  # type: ignore
            ContractError, artifact_ref, canonical_json, canonical_sha256,
            observe_runtime_evidence, runtime_evidence_sha256, validate_artifact_ref,
        )
        from desired_results_pair_source import PairSourceError, validate_pair_source  # type: ignore
        from desired_results_target import validate_target_manifest  # type: ignore
BUNDLE_INDEX_KEYS = frozenset({
    "schema_version", "bundle_kind", "target_id", "descriptor_sha256",
    "completion_index_ref", "target_manifest_ref",
})
COMPLETION_INDEX_KEYS = frozenset({
    "schema_version", "complete", "target_id", "descriptor_sha256",
    "activation_cache_sha256", "activation_record_sha256", "pair_texts_ref",
    "support_proof_ref", "route_count", "routes", "submission_performed",
    "model_loaded", "tensor_payload_downloaded",
})
SUPPORT_PROOF_KEYS = frozenset({
    "schema_version", "proof_kind", "target_id", "pair_count",
    "descriptor_sha256", "pair_text_hash", "pair_text_source",
    "split_algorithm", "split_counts", "splits", "support_sha256",
})
PAIR_TEXT_KEYS = frozenset({
    "schema_version", "target_id", "source", "pair_count", "pair_text_hash", "pairs",
})
ROUTE_PROOF_KEYS = frozenset({
    "schema_version", "proof_kind", "target_id", "activation_artifact", "route",
    "pair_ids", "tensor_shapes", "tensor_dtypes", "safetensors_header_length",
    "safetensors_header_sha256", "tensor_payload_downloaded",
})
ROUTE_COMPLETION_KEYS = frozenset({
    "schema_version", "complete", "target_id", "route", "proof_ref",
    "activation_lfs_sha256", "activation_header_sha256",
})
ARTIFACT_KEYS = frozenset({"repo_id", "repo_type", "revision", "path", "lfs_sha256", "size"})
HEX = frozenset("0123456789abcdef")
MAX_HEADER_BYTES = 16 * 1024 * 1024


class StaticAnalysisError(RuntimeError):
    """An immutable input or requested analysis violates the static contract."""


@dataclass(frozen=True)
class RouteData:
    target_id: str
    model: str
    benchmark: str
    strategy: str
    layer: int
    layer_count: int
    train_pair_ids: tuple[int, ...]
    train_stable_ids: tuple[str, ...]
    validation_pair_ids: tuple[int, ...]
    validation_stable_ids: tuple[str, ...]
    train_pos: np.ndarray
    train_neg: np.ndarray
    validation_pos: np.ndarray
    validation_neg: np.ndarray
    proof: Mapping[str, Any]
    completion: Mapping[str, Any]
    activation_path: Path
    completion_ref: Mapping[str, str] = field(default_factory=dict)
    proof_ref: Mapping[str, str] = field(default_factory=dict)

    @property
    def normalized_depth(self) -> float:
        return self.layer / self.layer_count


@dataclass(frozen=True)
class LoadedBundle:
    index_path: Path
    index: Mapping[str, Any]
    completion_index: Mapping[str, Any]
    target_manifest: Mapping[str, Any]
    support_proof: Mapping[str, Any]
    pair_texts_ref: Mapping[str, Any]
    routes: tuple[RouteData, ...]


# ---------- strict immutable input handling ----------


def _exact(value: Any, keys: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise StaticAnalysisError(f"{label} keys must be exactly {sorted(keys)}; got {actual}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise StaticAnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, label: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if type(value) is not int or value < minimum:
        raise StaticAnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise StaticAnalysisError(f"{label} must be a safe non-empty POSIX path")
    raw_parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in raw_parts):
        raise StaticAnalysisError(f"{label} must not be absolute or contain empty/dot traversal")
    path = PurePosixPath(value)
    return path


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise StaticAnalysisError(f"cannot read {label} {path}: {exc}") from exc


def _strict_json_bytes(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAnalysisError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise StaticAnalysisError(f"{label} must be a JSON object")
    return value


def _file_identity(path: Path) -> tuple[int, str]:
    raw = _read_bytes(path, "artifact")
    return len(raw), hashlib.sha256(raw).hexdigest()

def _local_file_ref(path: Path) -> dict[str, str]:
    size, digest = _file_identity(path)
    try:
        return artifact_ref(path.as_uri(), f"sha256:{digest}", str(size), digest)
    except ContractError as exc:
        raise StaticAnalysisError(f"invalid local ArtifactRef for {path}: {exc}") from exc


def _validate_ref(value: Any, label: str) -> Mapping[str, Any]:
    try:
        ref = validate_artifact_ref(value, label)
    except ContractError as exc:
        raise StaticAnalysisError(str(exc)) from exc
    digest = ref["sha256"]
    if ref["generation"] != f"sha256:{digest}":
        raise StaticAnalysisError(f"{label}.generation does not bind its SHA-256")
    return ref


def _bundle_ref_path(bundle_root: Path, value: Any, label: str) -> tuple[Path, Mapping[str, Any]]:
    ref = _validate_ref(value, label)
    prefix = "bundle:///"
    if not ref["uri"].startswith(prefix):
        raise StaticAnalysisError(f"{label}.uri must use bundle:/// identity")
    relative = _safe_relative(ref["uri"][len(prefix):], f"{label}.uri")
    path = bundle_root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(bundle_root.resolve())
    except (OSError, ValueError) as exc:
        raise StaticAnalysisError(f"{label}.uri escapes the immutable bundle") from exc
    return path, ref


def _resolve_bundle_ref(bundle_root: Path, value: Any, label: str) -> tuple[Path, Mapping[str, Any]]:
    path, ref = _bundle_ref_path(bundle_root, value, label)
    raw = _read_bytes(path, label)
    observed = hashlib.sha256(raw).hexdigest()
    if len(raw) != int(ref["size"]) or observed != ref["sha256"]:
        raise StaticAnalysisError(f"{label} byte identity differs from its immutable reference")
    return path, _strict_json_bytes(raw, label)


def _schema_two(value: Mapping[str, Any], label: str) -> None:
    if value.get("schema_version") != 2:
        raise StaticAnalysisError(f"{label}.schema_version must be 2")


def _validate_artifact(value: Any, label: str) -> Mapping[str, Any]:
    artifact = _exact(value, ARTIFACT_KEYS, label)
    for key in ("repo_id", "repo_type", "revision"):
        if not isinstance(artifact[key], str) or not artifact[key]:
            raise StaticAnalysisError(f"{label}.{key} must be a non-empty string")
    if artifact["repo_type"] not in {"dataset", "model", "space"}:
        raise StaticAnalysisError(f"{label}.repo_type is invalid")
    _safe_relative(artifact["path"], f"{label}.path")
    _sha(artifact["lfs_sha256"], f"{label}.lfs_sha256")
    _positive_int(artifact["size"], f"{label}.size")
    return artifact

def _validate_pair_source(value: Any, label: str) -> Mapping[str, Any]:
    try:
        return validate_pair_source(value, label)
    except PairSourceError as exc:
        raise StaticAnalysisError(str(exc)) from exc


def _activation_file(bundle_root: Path, cache_dir: Path, artifact: Mapping[str, Any]) -> Path:
    relative = _safe_relative(artifact["path"], "activation_artifact.path")
    candidates = [bundle_root.joinpath(*relative.parts), cache_dir.joinpath(*relative.parts)]
    for candidate in candidates:
        if candidate.is_file():
            size, digest = _file_identity(candidate)
            if size != artifact["size"] or digest != artifact["lfs_sha256"]:
                raise StaticAnalysisError(f"cached activation identity differs: {candidate}")
            return candidate

    destination = candidates[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    quoted_repo = urllib.parse.quote(artifact["repo_id"], safe="/")
    quoted_revision = urllib.parse.quote(artifact["revision"], safe="")
    quoted_path = urllib.parse.quote(artifact["path"], safe="/")
    repo_prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[artifact["repo_type"]]
    url = f"https://huggingface.co/{repo_prefix}{quoted_repo}/resolve/{quoted_revision}/{quoted_path}"
    request = urllib.request.Request(url, headers={"User-Agent": "wisent-static-analysis/1"})
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as stream:
            try:
                response = urllib.request.urlopen(request, timeout=120)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                raise StaticAnalysisError(f"cannot download pinned activation {artifact['path']}: {exc}") from exc
            with response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > artifact["size"]:
                        raise StaticAnalysisError("activation download exceeds the pinned byte size")
                    digest.update(chunk)
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if size != artifact["size"] or digest.hexdigest() != artifact["lfs_sha256"]:
            raise StaticAnalysisError("downloaded activation differs from pinned LFS identity")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
        observed_size, observed_sha = _file_identity(destination)
        if observed_size != size or observed_sha != digest.hexdigest():
            raise StaticAnalysisError("activation cache race produced a different immutable object")
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _safetensor_header(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise StaticAnalysisError(f"truncated safetensors prefix: {path}")
            length = struct.unpack("<Q", prefix)[0]
            if length <= 0 or length > MAX_HEADER_BYTES:
                raise StaticAnalysisError(f"invalid safetensors header length {length}: {path}")
            raw = stream.read(length)
            if len(raw) != length:
                raise StaticAnalysisError(f"truncated safetensors header: {path}")
    except OSError as exc:
        raise StaticAnalysisError(f"cannot read safetensors header {path}: {exc}") from exc
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StaticAnalysisError(f"invalid safetensors header JSON: {path}") from exc
    if not isinstance(header, Mapping):
        raise StaticAnalysisError(f"safetensors header must be an object: {path}")
    return header, raw


def _load_tensor_rows(path: Path, pair_ids: Sequence[int],
                      expected_pair_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Materialize only the requested pair rows from a pinned safetensors artifact."""
    requested = tuple(pair_ids)
    expected = tuple(expected_pair_ids)
    if (not requested or any(type(pair_id) is not int for pair_id in requested)
            or len(set(requested)) != len(requested)):
        raise StaticAnalysisError("requested activation pair IDs must be unique integers")
    if (any(type(pair_id) is not int for pair_id in expected)
            or len(set(expected)) != len(expected) or not set(requested) <= set(expected)):
        raise StaticAnalysisError("requested activation rows are not a subset of proven pair IDs")
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise StaticAnalysisError("safetensors and torch are required to load activation tensors") from exc
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if not {"pos_activations", "neg_activations"} <= keys:
                raise StaticAnalysisError("activation lacks pos_activations/neg_activations")
            metadata = handle.metadata() or {}
            try:
                metadata_pair_ids = json.loads(metadata["pair_ids"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise StaticAnalysisError("activation metadata lacks valid pair_ids") from exc
            if metadata_pair_ids != list(expected):
                raise StaticAnalysisError("activation pair_ids differ from proven support")
            positions = {pair_id: index for index, pair_id in enumerate(expected)}

            arrays: list[np.ndarray] = []
            for tensor_name in ("pos_activations", "neg_activations"):
                tensor_slice = handle.get_slice(tensor_name)
                shape = tensor_slice.get_shape()
                if len(shape) != 2 or shape[0] != len(expected) or shape[1] < 1:
                    raise StaticAnalysisError("activation tensor payload has invalid shape")
                # Index one proven row at a time. A wider slice could span a heldout row.
                selected = torch.cat([
                    tensor_slice[positions[pair_id]:positions[pair_id] + 1]
                    for pair_id in requested
                ], dim=0)
                array = selected.to(dtype=torch.float64).cpu().numpy()
                if not np.isfinite(array).all():
                    raise StaticAnalysisError("activation tensor payload contains non-finite selected rows")
                arrays.append(array)
    except StaticAnalysisError:
        raise
    except Exception as exc:
        raise StaticAnalysisError(f"cannot load pinned safetensors activation {path}: {exc}") from exc
    return arrays[0], arrays[1]


def _validated_splits(support: Mapping[str, Any], pair_count: int) -> tuple[dict[int, str], dict[int, str]]:
    expected_names = {"train", "validation", "test"}
    splits = support.get("splits")
    counts = support.get("split_counts")
    if not isinstance(splits, Mapping) or set(splits) != expected_names:
        raise StaticAnalysisError("support splits must contain exactly train, validation, and test")
    if not isinstance(counts, Mapping) or set(counts) != expected_names:
        raise StaticAnalysisError("support split_counts must contain exactly train, validation, and test")

    split_by_pair: dict[int, str] = {}
    stable_by_pair: dict[int, str] = {}
    stable_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        rows = splits[split]
        if not isinstance(rows, list) or not rows:
            raise StaticAnalysisError(f"support split {split} must be a non-empty list")
        if type(counts[split]) is not int or counts[split] != len(rows):
            raise StaticAnalysisError(f"support split_counts.{split} differs from its rows")
        for row in rows:
            row = _exact(row, {"pair_id", "stable_id"}, f"support {split} row")
            pair_id = _positive_int(row["pair_id"], "support pair_id", zero=True)
            stable_id = row["stable_id"]
            if pair_id >= pair_count:
                raise StaticAnalysisError("support pair_id is outside the declared support")
            if pair_id in split_by_pair:
                raise StaticAnalysisError("support splits overlap by pair_id")
            if not isinstance(stable_id, str) or not stable_id or stable_id in stable_ids:
                raise StaticAnalysisError("support stable identity is invalid or duplicated")
            split_by_pair[pair_id] = split
            stable_by_pair[pair_id] = stable_id
            stable_ids.add(stable_id)
    if set(split_by_pair) != set(range(pair_count)):
        raise StaticAnalysisError("support splits do not partition the exact declared pair IDs")
    return split_by_pair, stable_by_pair


def load_bundle(index_path: Path, cache_dir: Path) -> LoadedBundle:
    """Load and fully verify one immutable preflight bundle without loading a model."""
    index_path = Path(index_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    raw_index = _read_bytes(index_path, "bundle index")
    index = _exact(_strict_json_bytes(raw_index, "bundle index"), BUNDLE_INDEX_KEYS, "bundle index")
    _schema_two(index, "bundle index")
    if index["bundle_kind"] != "activation_preflight":
        raise StaticAnalysisError("bundle index has the wrong bundle_kind")
    if not isinstance(index["target_id"], str) or not index["target_id"]:
        raise StaticAnalysisError("bundle index target_id is invalid")
    descriptor_sha = _sha(index["descriptor_sha256"], "bundle index descriptor_sha256")
    # Content-addressed preflight indices must name their exact file bytes.
    expected_suffix = f".{hashlib.sha256(raw_index).hexdigest()}.json"
    if not index_path.name.endswith(expected_suffix):
        raise StaticAnalysisError("bundle index filename is not content-addressed by its bytes")
    root = index_path.parent
    _, completion = _resolve_bundle_ref(root, index["completion_index_ref"], "completion_index_ref")
    _, manifest = _resolve_bundle_ref(root, index["target_manifest_ref"], "target_manifest_ref")
    completion = _exact(completion, COMPLETION_INDEX_KEYS, "completion index")
    _schema_two(completion, "completion index")
    try:
        validate_target_manifest(manifest)
    except Exception as exc:
        raise StaticAnalysisError(f"TargetManifest validation failed: {exc}") from exc
    if manifest["manifest_sha256"] != canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise StaticAnalysisError("TargetManifest canonical identity differs")
    target_id = index["target_id"]
    if (completion["target_id"] != target_id or manifest["target"]["target_id"] != target_id or
            completion["descriptor_sha256"] != descriptor_sha):
        raise StaticAnalysisError("bundle target or descriptor identity differs")
    if completion["complete"] is not True or any(completion[name] is not False for name in (
            "submission_performed", "model_loaded", "tensor_payload_downloaded")):
        raise StaticAnalysisError("completion index is not a CPU-only complete preflight")
    _sha(completion["activation_cache_sha256"], "activation_cache_sha256")
    _sha(completion["activation_record_sha256"], "activation_record_sha256")

    _, support = _resolve_bundle_ref(root, completion["support_proof_ref"], "support_proof_ref")
    support = _exact(support, SUPPORT_PROOF_KEYS, "support proof")
    _schema_two(support, "support proof")
    if support["target_id"] != target_id:
        raise StaticAnalysisError("support target identity differs")
    pair_count = manifest["target"]["expected_pairs"]
    if support["pair_count"] != pair_count:
        raise StaticAnalysisError("support count differs from TargetManifest")
    if support["descriptor_sha256"] != descriptor_sha:
        raise StaticAnalysisError("support proof descriptor identity differs")
    _sha(support["pair_text_hash"], "support pair_text_hash")
    _validate_pair_source(support["pair_text_source"], "support pair_text_source")
    if support["support_sha256"] != canonical_sha256(support["splits"]):
        raise StaticAnalysisError("support split proof hash differs")
    if completion["support_proof_ref"]["sha256"] != manifest["support"]["proof_sha256"]:
        raise StaticAnalysisError("support proof ArtifactRef differs from TargetManifest")
    if completion["pair_texts_ref"] != manifest["support"]["pair_texts_ref"]:
        raise StaticAnalysisError("pair-text ArtifactRef differs from TargetManifest")
    if support["splits"] != manifest["support"]["splits"] or support["split_counts"] != manifest["support"]["split_counts"]:
        raise StaticAnalysisError("support splits differ from TargetManifest")
    # Pair text is not an input to static fitting or ranking. Validate its immutable
    # reference and confinement, but deliberately never read its mixed-split payload.
    _, pair_texts_ref = _bundle_ref_path(root, completion["pair_texts_ref"], "pair_texts_ref")
    split_by_pair, stable_by_pair = _validated_splits(support, pair_count)

    routes_value = completion["routes"]
    if not isinstance(routes_value, list) or completion["route_count"] != len(routes_value):
        raise StaticAnalysisError("completion route_count differs from routes")
    manifest_routes = {(r["strategy"], r["layer"]): r for r in manifest["activation"]["routes"]}
    expected_routes = set(manifest_routes)
    if len(routes_value) != len(expected_routes):
        raise StaticAnalysisError("completion does not contain the TargetManifest route matrix")
    layer_count = manifest["activation"]["layer_count"]
    model = manifest["target"]["model_name"]
    benchmark = manifest["target"]["benchmark"]
    activation_revision = manifest["revisions"]["activation_revision"]
    loaded_routes: list[RouteData] = []
    seen: set[tuple[str, int]] = set()
    for route_index, route_value in enumerate(routes_value):
        route = _exact(route_value, {"strategy", "layer", "completion_ref", "proof_ref"}, f"routes[{route_index}]")
        key = (route["strategy"], route["layer"])
        if key not in expected_routes or key in seen:
            raise StaticAnalysisError("completion route matrix differs from TargetManifest")
        target_route = manifest_routes[key]
        if (route["completion_ref"] != target_route["completion_ref"] or
                route["proof_ref"] != target_route["proof_ref"]):
            raise StaticAnalysisError("completion route refs differ from TargetManifest")
        seen.add(key)
        _, route_completion = _resolve_bundle_ref(root, route["completion_ref"], f"routes[{route_index}].completion_ref")
        _, proof = _resolve_bundle_ref(root, route["proof_ref"], f"routes[{route_index}].proof_ref")
        route_completion = _exact(route_completion, ROUTE_COMPLETION_KEYS, "route completion")
        proof = _exact(proof, ROUTE_PROOF_KEYS, "route proof")
        _schema_two(route_completion, "route completion")
        _schema_two(proof, "route proof")
        route_identity = {"strategy": key[0], "layer": key[1]}
        if (route_completion["complete"] is not True or route_completion["target_id"] != target_id or
                proof["target_id"] != target_id or route_completion["route"] != route_identity or
                proof["route"] != route_identity or route_completion["proof_ref"] != route["proof_ref"]):
            raise StaticAnalysisError("route completion/proof identity differs")
        artifact = _validate_artifact(proof["activation_artifact"], "activation_artifact")
        if artifact["revision"] != activation_revision:
            raise StaticAnalysisError("activation artifact revision differs from TargetManifest")
        if route_completion["activation_lfs_sha256"] != artifact["lfs_sha256"]:
            raise StaticAnalysisError("route completion LFS identity differs")
        if route_completion["activation_header_sha256"] != proof["safetensors_header_sha256"]:
            raise StaticAnalysisError("route completion header identity differs")
        if proof["tensor_payload_downloaded"] is not False:
            raise StaticAnalysisError("preflight route unexpectedly claims tensor payload download")
        activation_path = _activation_file(root, cache_dir, artifact)
        header, raw_header = _safetensor_header(activation_path)
        if (len(raw_header) != proof["safetensors_header_length"] or
                hashlib.sha256(raw_header).hexdigest() != proof["safetensors_header_sha256"]):
            raise StaticAnalysisError("activation header differs from route proof")
        shapes = proof["tensor_shapes"]
        dtypes = proof["tensor_dtypes"]
        if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
            raise StaticAnalysisError("route proof tensor header maps are invalid")
        header_shapes = {name: header.get(name, {}).get("shape") for name in ("pos_activations", "neg_activations")}
        header_dtypes = {name: header.get(name, {}).get("dtype") for name in ("pos_activations", "neg_activations")}
        if (dict(shapes) != header_shapes or dict(dtypes) != header_dtypes
                or shapes.get("pos_activations") != shapes.get("neg_activations")
                or not isinstance(shapes.get("pos_activations"), list)
                or len(shapes["pos_activations"]) != 2
                or shapes["pos_activations"][0] != pair_count):
            raise StaticAnalysisError("proven activation tensor headers have invalid shape or dtype")
        proven_ids = proof["pair_ids"]
        if (not isinstance(proven_ids, list) or len(proven_ids) != pair_count
                or any(type(pair_id) is not int for pair_id in proven_ids)
                or set(proven_ids) != set(split_by_pair)):
            raise StaticAnalysisError("activation pair_ids differ from proven support")
        train_ids = tuple(pair_id for pair_id in proven_ids if split_by_pair[pair_id] == "train")
        validation_ids = tuple(pair_id for pair_id in proven_ids if split_by_pair[pair_id] == "validation")
        train_pos, train_neg = _load_tensor_rows(activation_path, train_ids, proven_ids)
        validation_pos, validation_neg = _load_tensor_rows(activation_path, validation_ids, proven_ids)
        loaded_routes.append(RouteData(
            target_id=target_id, model=model, benchmark=benchmark, strategy=key[0], layer=key[1],
            layer_count=layer_count,
            train_pair_ids=train_ids,
            train_stable_ids=tuple(stable_by_pair[pair_id] for pair_id in train_ids),
            validation_pair_ids=validation_ids,
            validation_stable_ids=tuple(stable_by_pair[pair_id] for pair_id in validation_ids),
            train_pos=train_pos, train_neg=train_neg,
            validation_pos=validation_pos, validation_neg=validation_neg,
            proof=proof, completion=route_completion, completion_ref=dict(route["completion_ref"]),
            proof_ref=dict(route["proof_ref"]), activation_path=activation_path,
        ))
    if seen != expected_routes:
        raise StaticAnalysisError("not every TargetManifest route was loaded")
    loaded_routes.sort(key=lambda route: (route.strategy, route.layer))
    return LoadedBundle(index_path, index, completion, manifest, support, dict(pair_texts_ref), tuple(loaded_routes))


# ---------- deterministic statistics ----------


def _seed64(seed: Any) -> int:
    if type(seed) is not int or seed < 0 or seed >= 1 << 64:
        raise StaticAnalysisError("seed must be an unsigned 64-bit integer")
    return seed


def _sklearn_seed(seed: int) -> int:
    """Map the public uint64 seed deterministically into sklearn's uint32 domain."""
    seed = _seed64(seed)
    return int.from_bytes(hashlib.sha256(seed.to_bytes(8, "big")).digest()[:4], "big")


def _derived_seed(seed: int, *parts: Any) -> int:
    material = seed.to_bytes(8, "big") + b"\0" + b"\0".join(str(part).encode("utf-8") for part in parts)
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")





def _validate_arrays(pos: Any, neg: Any, pair_ids: Sequence[int], *, label: str,
                     minimum_pairs: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    pos_array = np.asarray(pos, dtype=np.float64)
    neg_array = np.asarray(neg, dtype=np.float64)
    ids = tuple(pair_ids)
    if pos_array.ndim != 2 or neg_array.ndim != 2 or pos_array.shape != neg_array.shape:
        raise StaticAnalysisError(f"{label} positive and negative activations must be equal two-dimensional arrays")
    if pos_array.shape[0] != len(ids) or pos_array.shape[0] < minimum_pairs or pos_array.shape[1] < 1:
        raise StaticAnalysisError(f"{label} activation rows do not satisfy the declared pair support")
    if any(type(pair_id) is not int for pair_id in ids) or len(set(ids)) != len(ids):
        raise StaticAnalysisError(f"{label} pair IDs must be unique integers")
    if not np.isfinite(pos_array).all() or not np.isfinite(neg_array).all():
        raise StaticAnalysisError(f"{label} activation arrays must contain only finite values")
    return pos_array, neg_array, ids


def _balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        raise StaticAnalysisError("balanced accuracy requires both classes")
    return float(((predicted[positive] == 1).mean() + (predicted[negative] == 0).mean()) / 2)


def _auroc(y: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        result = float(roc_auc_score(y, scores))
    except ValueError as exc:
        raise StaticAnalysisError(f"AUROC is undefined for a supposedly valid paired fold: {exc}") from exc
    if not math.isfinite(result):
        raise StaticAnalysisError("AUROC returned a non-finite result")
    return result


def _fit_scores(train_pos: np.ndarray, train_neg: np.ndarray, validation_pos: np.ndarray,
                validation_neg: np.ndarray, seed: int) -> dict[str, float]:
    train_x = np.concatenate((train_pos, train_neg), axis=0)
    train_y = np.concatenate((np.ones(len(train_pos), dtype=np.int8), np.zeros(len(train_neg), dtype=np.int8)))
    validation_x = np.concatenate((validation_pos, validation_neg), axis=0)
    validation_y = np.concatenate((np.ones(len(validation_pos), dtype=np.int8), np.zeros(len(validation_neg), dtype=np.int8)))

    pos_centroid = train_pos.mean(axis=0)
    neg_centroid = train_neg.mean(axis=0)
    direction = pos_centroid - neg_centroid
    midpoint = (pos_centroid + neg_centroid) / 2
    centroid_scores = (validation_x - midpoint) @ direction
    centroid_pred = (centroid_scores >= 0).astype(np.int8)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train_x)
        scaled_train = scaler.transform(train_x)
        scaled_validation = scaler.transform(validation_x)
        probe = LogisticRegression(
            random_state=_sklearn_seed(seed), solver="liblinear", penalty="l2", C=1.0,
            max_iter=2000,
        ).fit(scaled_train, train_y)
        probe_scores = probe.decision_function(scaled_validation)
        probe_pred = probe.predict(scaled_validation)
    except Exception as exc:
        raise StaticAnalysisError(f"logistic probe failed on the fixed train/validation split: {exc}") from exc
    return {
        "centroid_balanced_accuracy": _balanced_accuracy(validation_y, centroid_pred),
        "centroid_auroc": _auroc(validation_y, centroid_scores),
        "probe_balanced_accuracy": _balanced_accuracy(validation_y, probe_pred),
        "probe_auroc": _auroc(validation_y, probe_scores),
    }


def analyze_route(train_pos: Any, train_neg: Any, train_pair_ids: Sequence[int],
                  validation_pos: Any, validation_neg: Any,
                  validation_pair_ids: Sequence[int], *, seed: int,
                  bootstrap_count: int, permutation_count: int) -> dict[str, Any]:
    """Fit/statistic on train and score a fixed route on validation only."""
    train_pos_array, train_neg_array, train_ids = _validate_arrays(
        train_pos, train_neg, train_pair_ids, label="train", minimum_pairs=2,
    )
    validation_pos_array, validation_neg_array, validation_ids = _validate_arrays(
        validation_pos, validation_neg, validation_pair_ids,
        label="validation", minimum_pairs=1,
    )
    if train_pos_array.shape[1] != validation_pos_array.shape[1]:
        raise StaticAnalysisError("train and validation hidden sizes differ")
    if set(train_ids) & set(validation_ids):
        raise StaticAnalysisError("train and validation pair IDs overlap")
    seed = _seed64(seed)
    if type(bootstrap_count) is not int or bootstrap_count < 1:
        raise StaticAnalysisError("bootstrap_count must be a positive integer")
    if type(permutation_count) is not int or permutation_count < 1:
        raise StaticAnalysisError("permutation_count must be a positive integer")

    differences = train_pos_array - train_neg_array
    mean_difference = differences.mean(axis=0)
    separation = float(np.linalg.norm(mean_difference))
    if separation == 0.0:
        projected = np.zeros(len(train_ids), dtype=np.float64)
        effect: float | None = 0.0
        effect_status = "zero_direction"
    else:
        projected = differences @ (mean_difference / separation)
        deviation = float(projected.std(ddof=1))
        if deviation == 0.0:
            effect = None
            effect_status = "unbounded_constant_projection"
        else:
            effect = float(projected.mean() / deviation)
            effect_status = "ok"

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_count, dtype=np.float64)
    for index in range(bootstrap_count):
        sample = rng.integers(0, len(projected), size=len(projected))
        bootstrap[index] = projected[sample].mean()
    ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
    observed = abs(float(projected.mean()))
    exceedances = 0
    for _ in range(permutation_count):
        signs = rng.integers(0, 2, size=len(projected), dtype=np.int8) * 2 - 1
        exceedances += abs(float(np.mean(projected * signs))) >= observed
    sign_flip_p = (exceedances + 1) / (permutation_count + 1)

    validation_metrics = _fit_scores(
        train_pos_array, train_neg_array, validation_pos_array, validation_neg_array,
        _derived_seed(seed, "probe"),
    )
    return {
        "status": "ok" if effect_status == "ok" else effect_status,
        "train_pair_count": len(train_ids),
        "validation_pair_count": len(validation_ids),
        "hidden_size": train_pos_array.shape[1],
        "paired_mean_difference_norm": separation,
        "paired_projected_mean": float(projected.mean()),
        "paired_standardized_effect": effect,
        "paired_standardized_effect_status": effect_status,
        "bootstrap_ci_95": {"low": float(ci_low), "high": float(ci_high), "replicates": bootstrap_count},
        "sign_flip_test": {"p_value": float(sign_flip_p), "permutations": permutation_count,
                           "alternative": "two-sided"},
        **validation_metrics,
    }


# ---------- report construction ----------


def _route_identity(route: RouteData) -> dict[str, Any]:
    return {
        "target_id": route.target_id, "model": route.model, "benchmark": route.benchmark,
        "strategy": route.strategy, "layer": route.layer,
        "normalized_depth": route.normalized_depth,
    }


def _support_hash(pair_ids: Sequence[int]) -> str:
    return canonical_sha256(list(pair_ids))


def _sample_sizes(maximum: int) -> tuple[int, ...]:
    if maximum < 2:
        return ()
    values: list[int] = []
    size = 2
    while size < maximum:
        values.append(size)
        size *= 2
    values.append(maximum)
    return tuple(dict.fromkeys(values))


def _sample_efficiency(route: RouteData, seed: int) -> list[dict[str, Any]]:
    id_to_index = {pair_id: index for index, pair_id in enumerate(route.train_pair_ids)}
    ranked_train = sorted(route.train_pair_ids, key=lambda pair_id: (
        hashlib.sha256(
            _derived_seed(seed, route.target_id, route.strategy, route.layer).to_bytes(8, "big")
            + b"\0" + route.train_stable_ids[id_to_index[pair_id]].encode("utf-8")
        ).digest(),
        pair_id,
    ))
    rows: list[dict[str, Any]] = []
    for sample_size in _sample_sizes(len(ranked_train)):
        selected_ids = tuple(sorted(ranked_train[:sample_size]))
        if set(selected_ids) & set(route.validation_pair_ids):
            raise StaticAnalysisError("sample-efficiency train and validation pair groups overlap")
        train_indices = np.array([id_to_index[pair_id] for pair_id in selected_ids], dtype=np.int64)
        metrics = _fit_scores(
            route.train_pos[train_indices], route.train_neg[train_indices],
            route.validation_pos, route.validation_neg,
            _derived_seed(seed, "sample", route.target_id, route.strategy, route.layer, sample_size),
        )
        rows.append({
            "schema_version": 1, **_route_identity(route),
            "sample_size_pairs": sample_size,
            "validation_size_pairs": len(route.validation_pair_ids),
            "train_pair_ids_sha256": _support_hash(selected_ids),
            "validation_pair_ids_sha256": _support_hash(route.validation_pair_ids),
            **metrics,
        })
    return rows


def _summary_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_target: dict[str, list[Mapping[str, Any]]] = {}
    for metric in metrics:
        by_target.setdefault(str(metric["target_id"]), []).append(metric)
    for target_id in sorted(by_target):
        target_rows = by_target[target_id]
        best_by_strategy: dict[str, tuple[float, int]] = {}
        best_by_layer: dict[int, tuple[float, str]] = {}
        for metric in target_rows:
            strategy = str(metric["strategy"])
            layer = int(metric["layer"])
            score = float(metric["probe_balanced_accuracy"])
            candidate_layer = (score, -layer)
            incumbent_layer = best_by_strategy.get(strategy)
            if incumbent_layer is None or candidate_layer > (incumbent_layer[0], -incumbent_layer[1]):
                best_by_strategy[strategy] = (score, layer)
            candidate_strategy = (score, strategy)
            incumbent_strategy = best_by_layer.get(layer)
            if incumbent_strategy is None or score > incumbent_strategy[0] or (score == incumbent_strategy[0] and strategy < incumbent_strategy[1]):
                best_by_layer[layer] = candidate_strategy
        overall = min(target_rows, key=lambda row: (-float(row["probe_balanced_accuracy"]), int(row["layer"]), str(row["strategy"])))
        for metric in sorted(target_rows, key=lambda row: (str(row["strategy"]), int(row["layer"]))):
            strategy = str(metric["strategy"])
            layer = int(metric["layer"])
            rows.append({
                "schema_version": 1,
                **{key: metric[key] for key in ("target_id", "model", "benchmark", "strategy", "layer", "normalized_depth")},
                "probe_balanced_accuracy": metric["probe_balanced_accuracy"],
                "probe_auroc": metric["probe_auroc"],
                "paired_mean_difference_norm": metric["paired_mean_difference_norm"],
                "best_layer_for_strategy": layer == best_by_strategy[strategy][1],
                "best_strategy_for_layer": strategy == best_by_layer[layer][1],
                "best_overall": strategy == overall["strategy"] and layer == overall["layer"],
            })
    return rows


def _standardized_transfer(source: RouteData, target: RouteData, seed: int) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "source_target_id": source.target_id, "source_model": source.model,
        "source_benchmark": source.benchmark, "source_strategy": source.strategy,
        "source_layer": source.layer, "source_normalized_depth": source.normalized_depth,
        "target_target_id": target.target_id, "target_model": target.model,
        "target_benchmark": target.benchmark, "target_strategy": target.strategy,
        "target_layer": target.layer, "target_normalized_depth": target.normalized_depth,
    }
    source_hidden_size = source.train_pos.shape[1]
    target_hidden_size = target.train_pos.shape[1]
    if source_hidden_size != target_hidden_size:
        return {**identity, "status": "incompatible_hidden_size",
                "source_hidden_size": source_hidden_size,
                "target_hidden_size": target_hidden_size,
                "metrics": {"status": "unavailable_incompatible_hidden_size"}}
    source_train = np.concatenate((source.train_pos, source.train_neg))
    target_train = np.concatenate((target.train_pos, target.train_neg))
    source_mean, source_scale = source_train.mean(axis=0), source_train.std(axis=0)
    target_mean, target_scale = target_train.mean(axis=0), target_train.std(axis=0)
    source_scale[source_scale == 0] = 1.0
    target_scale[target_scale == 0] = 1.0
    source_pos = (source.train_pos - source_mean) / source_scale
    source_neg = (source.train_neg - source_mean) / source_scale
    target_validation_pos = (target.validation_pos - target_mean) / target_scale
    target_validation_neg = (target.validation_neg - target_mean) / target_scale
    metrics = _fit_scores(
        source_pos, source_neg, target_validation_pos, target_validation_neg, seed,
    )
    return {**identity, "status": "ok", "source_hidden_size": source_hidden_size,
            "target_hidden_size": target_hidden_size, "metrics": metrics}


def _transfer_rows(routes: Sequence[RouteData], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RouteData]] = {}
    for route in routes:
        grouped.setdefault((route.target_id, route.strategy), []).append(route)
    rows: list[dict[str, Any]] = []
    targets = sorted({route.target_id for route in routes})
    strategies = sorted({route.strategy for route in routes})
    for source_target in targets:
        for target_target in targets:
            if source_target == target_target:
                continue
            for strategy in strategies:
                sources = sorted(grouped.get((source_target, strategy), ()), key=lambda route: route.layer)
                targets_for_strategy = grouped.get((target_target, strategy), ())
                if not sources or not targets_for_strategy:
                    continue
                for source in sources:
                    target = min(targets_for_strategy, key=lambda route: (abs(route.normalized_depth - source.normalized_depth), route.layer))
                    rows.append(_standardized_transfer(
                        source, target, _derived_seed(seed, "transfer", source.target_id, target.target_id,
                                                      strategy, source.layer, target.layer),
                    ))
    return rows


def _canonical_lines(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def _write_addressed(stage: Path, stem: str, suffix: str, raw: bytes) -> tuple[Path, dict[str, str]]:
    digest = hashlib.sha256(raw).hexdigest()
    name = f"{stem}.{digest}.{suffix}"
    path = stage / name
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise StaticAnalysisError(f"cannot create analysis artifact {path}: {exc}") from exc
    try:
        ref = artifact_ref(f"analysis:///{name}", f"sha256:{digest}", str(len(raw)), digest)
    except ContractError as exc:
        raise StaticAnalysisError(f"invalid output ArtifactRef: {exc}") from exc
    return path, ref


def run(bundle_indexes: Sequence[Path], cache_dir: Path, output_dir: Path, *, seed: int = 0, bootstrap_count: int = 1000, permutation_count: int = 1000) -> Path:
    """Analyze immutable bundles and atomically create a content-addressed report."""
    seed = _seed64(seed)
    if not bundle_indexes:
        raise StaticAnalysisError("at least one --bundle-index is required")
    resolved_indexes = [Path(path).resolve() for path in bundle_indexes]
    if len(resolved_indexes) != len(set(resolved_indexes)):
        raise StaticAnalysisError("bundle indexes must be unique")
    output_dir = Path(output_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    if output_dir.exists():
        raise StaticAnalysisError(f"output destination already exists: {output_dir}")
    bundles = [load_bundle(path, cache_dir) for path in resolved_indexes]
    if len({bundle.index["target_id"] for bundle in bundles}) != len(bundles):
        raise StaticAnalysisError("bundle target IDs must be unique")
    bundles.sort(key=lambda bundle: str(bundle.index["target_id"]))
    routes = [route for bundle in bundles for route in bundle.routes]
    if not routes:
        raise StaticAnalysisError("immutable bundles contain no activation routes")

    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for route in routes:
        route_seed = _derived_seed(seed, route.target_id, route.strategy, route.layer)
        metrics = analyze_route(
            route.train_pos, route.train_neg, route.train_pair_ids,
            route.validation_pos, route.validation_neg, route.validation_pair_ids,
            seed=route_seed, bootstrap_count=bootstrap_count,
            permutation_count=permutation_count,
        )
        metric_rows.append({
            "schema_version": 1, **_route_identity(route),
            "train_support_sha256": _support_hash(route.train_pair_ids),
            "validation_support_sha256": _support_hash(route.validation_pair_ids),
            **metrics,
        })
        
        sample_rows.extend(_sample_efficiency(route, route_seed))
    metric_rows.sort(key=lambda row: (str(row["target_id"]), str(row["strategy"]), int(row["layer"])))
    summary_rows = _summary_rows(metric_rows)
    transfer_rows = _transfer_rows(routes, seed)
    if not transfer_rows:
        transfer_rows = [{"schema_version": 1, "status": "unavailable_requires_multiple_targets",
                          "target_count": len(bundles)}]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _, metrics_ref = _write_addressed(stage, "metrics", "jsonl", _canonical_lines(metric_rows))
        _, summary_ref = _write_addressed(stage, "layer-format-summary", "jsonl", _canonical_lines(summary_rows))
        _, sample_ref = _write_addressed(stage, "sample-efficiency", "jsonl", _canonical_lines(sample_rows))
        _, transfer_ref = _write_addressed(stage, "transfer-matrix", "jsonl", _canonical_lines(transfer_rows))
        implementation_ref = _local_file_ref(Path(__file__).resolve())
        runtime_evidence = observe_runtime_evidence()
        provenance = {
            "schema_version": 1, "analysis": "desired_results_static_analysis", "seed_uint64": seed,
            "sklearn_seed_mapping": "sha256(uint64-big-endian) first 32 bits",
            "split_mode": "fixed_train_validation", "bootstrap_count": bootstrap_count, "permutation_count": permutation_count,
            "implementation_ref": implementation_ref,
            "runtime_evidence": runtime_evidence,
            "runtime_evidence_sha256": runtime_evidence_sha256(runtime_evidence),
            "analysis_dependencies": {
                "numpy": importlib.metadata.version("numpy"),
                "scikit-learn": importlib.metadata.version("scikit-learn"),
                "safetensors": importlib.metadata.version("safetensors"),
            },
            "bundles": [{
                "target_id": bundle.index["target_id"],
                "bundle_index_ref": _local_file_ref(bundle.index_path),
                "descriptor_sha256": bundle.index["descriptor_sha256"],
                "target_manifest_sha256": bundle.target_manifest["manifest_sha256"],
                "revisions": bundle.target_manifest["revisions"],
                "routes": [{"strategy": route.strategy, "layer": route.layer,
                            "completion_ref": route.completion_ref,
                            "proof_ref": route.proof_ref,
                            "activation_artifact": route.proof["activation_artifact"]}
                           for route in bundle.routes],
            } for bundle in bundles],
        }
        _, provenance_ref = _write_addressed(stage, "provenance", "json", canonical_json(provenance) + b"\n")
        artifacts = {
            "metrics": metrics_ref, "layer_format_summary": summary_ref,
            "sample_efficiency": sample_ref, "transfer_matrix": transfer_ref,
            "provenance": provenance_ref,
        }
        index = {
            "schema_version": 1, "analysis": "desired_results_static_analysis", "complete": True,
            "target_ids": sorted(bundle.index["target_id"] for bundle in bundles),
            "route_count": len(routes), "artifacts": artifacts,
            "provenance_sha256": provenance_ref["sha256"],
        }
        index_raw = canonical_json(index) + b"\n"
        index_path, _ = _write_addressed(stage, "static-analysis-index", "json", index_raw)
        try:
            os.replace(stage, output_dir)
        except OSError as exc:
            raise StaticAnalysisError(f"cannot publish create-only analysis directory: {exc}") from exc
        return output_dir / index_path.name
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-index", action="append", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--permutation-count", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        index = run(
            args.bundle_index, args.cache_dir, args.output_dir, seed=args.seed,
            bootstrap_count=args.bootstrap_count,
            permutation_count=args.permutation_count,
        )
    except StaticAnalysisError as exc:
        print(f"static analysis failed: {exc}", file=__import__("sys").stderr)
        return 2
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
