#!/usr/bin/env python3
"""Exact CalibrationManifestV3 execution engine.

This module is deliberately a library adapter.  The Stado-facing worker owns
attempt claims and terminal receipts; this runner owns immutable input
resolution, bounded fresh Optuna execution, and create-only result publication.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import os
import struct
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_sibling(name: str):
    try:
        return __import__(f"scripts.steering.{name}", fromlist=[name])
    except (ImportError, ModuleNotFoundError):
        path = Path(__file__).with_name(f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


contract = _load_sibling("desired_results_execution_contract")
ContractError = contract.ContractError
METHODS = contract.desired_results_target.METHODS
STRATEGIES = (
    "chat_first", "chat_last", "chat_max_norm", "chat_mean",
    "chat_weighted", "mc_balanced", "role_play",
)
_DIMENSION_KEYS = frozenset({
    "hidden_dim", "gate_hidden_dim", "intensity_hidden_dim", "flow_hidden_dim",
    "concept_dim", "max_concept_dim", "gate_dim_min", "gate_dim_max",
    "intensity_dim_min", "intensity_dim_max", "num_dims", "num_directions",
})
_ORDERED_CONDITIONS = {
    "tecza": (("num_directions", "max_directions", False),
              ("min_cosine_similarity", "max_cosine_similarity", False)),
    "tetno": (("entropy_floor", "entropy_ceiling", True),
              ("steering_start", "steering_end", False)),
    "grom": (("warmup_steps", "optimization_steps", True),
             ("steering_start", "steering_end", False),
             ("min_cosine_sim", "max_cosine_sim", False),
             ("gate_dim_min", "gate_dim_max", False),
             ("intensity_dim_min", "intensity_dim_max", False),
             ("adapt_linear_directions", "adapt_complex_directions", False),
             ("adapt_complex_directions", "adapt_max_directions", False)),
    "nurt": (("num_dims", "max_concept_dim", False), ("lr_min", "lr", False)),
}


class GCSStore:
    """Minimal immutable object API used by Stado calibration workers."""
    def __init__(self) -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ContractError("google-cloud-storage is required") from exc
        self._client = storage.Client()

    @staticmethod
    def _parts(uri: str) -> tuple[str, str]:
        if not isinstance(uri, str) or not uri.startswith("gs://"):
            raise ContractError(f"not a GCS URI: {uri!r}")
        bucket, slash, name = uri[5:].partition("/")
        if not bucket or not slash or not name:
            raise ContractError(f"incomplete GCS URI: {uri!r}")
        return bucket, name

    def exists(self, uri: str) -> bool:
        bucket, name = self._parts(uri)
        return bool(self._client.bucket(bucket).blob(name).exists())

    def create(self, uri: str, data: bytes, content_type: str = "application/json") -> dict[str, str]:
        bucket, name = self._parts(uri)
        blob = self._client.bucket(bucket).blob(name)
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
            blob.reload()
        except Exception as exc:
            raise ContractError(f"create-only upload refused for {uri}: {exc}") from exc
        return contract.artifact_ref(uri, str(blob.generation), str(len(data)), hashlib.sha256(data).hexdigest())

    def read(self, uri: str, generation: str | None = None) -> tuple[bytes, str]:
        bucket, name = self._parts(uri)
        blob = self._client.bucket(bucket).blob(
            name, generation=int(generation) if generation is not None else None,
        )
        try:
            if generation is not None:
                data = blob.download_as_bytes(if_generation_match=int(generation))
                observed = str(generation)
            else:
                blob.reload()
                observed = str(blob.generation)
                data = blob.download_as_bytes(if_generation_match=int(observed))
        except Exception as exc:
            raise ContractError(f"cannot read immutable object {uri}: {exc}") from exc
        return data, observed


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    if contract.canonical_json(value) != data:
        raise ContractError(f"{label} is not canonical JSON")
    return value


def _read_ref(store: Any, ref_value: Mapping[str, Any], label: str) -> dict[str, Any]:
    ref = contract.validate_artifact_ref(ref_value, label)
    data, generation = store.read(ref["uri"], ref["generation"])
    if str(generation) != ref["generation"]:
        raise ContractError(f"{label} generation drift")
    document = _json_object(data, label)
    contract.validate_artifact_binding(ref, document, label)
    return document


def _load_manifest(store: Any, uri: str, generation: str) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    data, observed = store.read(uri, generation)
    if str(observed) != str(generation):
        raise ContractError("calibration manifest generation drift")
    manifest = _json_object(data, "calibration manifest")
    manifest_ref = contract.artifact_ref(uri, str(generation), str(len(data)), hashlib.sha256(data).hexdigest())
    contract.validate_artifact_binding(manifest_ref, manifest, "calibration manifest")
    target_manifest = _read_ref(store, manifest["target_manifest_ref"], "target manifest")
    contract.validate_calibration_manifest(manifest, target_manifest)
    return manifest, manifest_ref, target_manifest


def _create_once(store: Any, uri: str, document: Mapping[str, Any]) -> dict[str, str]:
    data = contract.canonical_json(document)
    digest = hashlib.sha256(data).hexdigest()
    if store.exists(uri):
        observed, generation = store.read(uri)
        if observed != data:
            raise ContractError(f"conflicting immutable calibration object: {uri}")
        return contract.artifact_ref(uri, str(generation), str(len(data)), digest)
    try:
        ref = store.create(uri, data, content_type="application/json")
    except Exception:
        if not store.exists(uri):
            raise
        observed, generation = store.read(uri)
        if observed != data:
            raise ContractError(f"conflicting immutable calibration create race: {uri}")
        ref = {"uri": uri, "generation": str(generation), "size": str(len(data)), "sha256": digest}
    normalized = contract.validate_artifact_ref(ref, "calibration result ref")
    if normalized["uri"] != uri or normalized["size"] != str(len(data)) or normalized["sha256"] != digest:
        raise ContractError("store returned incorrect calibration result ArtifactRef")
    readback, generation = store.read(uri, normalized["generation"])
    if readback != data or str(generation) != normalized["generation"]:
        raise ContractError("calibration result failed immutable read-back")
    return normalized


def _canonical_param(param: Any) -> dict[str, Any]:
    from wisent.core.utils.services.optimization.core.parameters import (
        CategoricalParam, FloatParam, IntParam,
    )
    if isinstance(param, CategoricalParam):
        if not isinstance(param.choices, list) or not param.choices:
            raise ContractError("categorical parameter has empty support")
        return {"kind": "categorical", "choices": list(param.choices)}
    if isinstance(param, FloatParam):
        if param.distribution not in {"normal", "lognormal", "uniform"}:
            raise ContractError(f"unsupported float distribution {param.distribution!r}")
        result = {
            "kind": "float", "distribution": param.distribution,
            "mu": param.mu, "sigma": param.sigma, "low": param.low,
            "high": param.high, "log_scale": bool(param.log_scale),
        }
    elif isinstance(param, IntParam):
        if param.distribution not in {"randint", "qnormal", "qlognormal"}:
            raise ContractError(f"unsupported integer distribution {param.distribution!r}")
        result = {
            "kind": "int", "distribution": param.distribution,
            "mu": param.mu, "sigma": param.sigma, "q": param.q,
            "low": param.low, "high": param.high,
        }
    else:
        raise ContractError(f"unsupported parameter type {type(param).__name__}")
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError(f"parameter field {key} is not finite")
    return result


def _serialize_space(space: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: _canonical_param(space[name]) for name in sorted(space)}


def _param_from_spec(spec: Mapping[str, Any]):
    from wisent.core.utils.services.optimization.core.parameters import (
        CategoricalParam, FloatParam, IntParam,
    )
    kind = spec.get("kind")
    if kind == "categorical":
        return CategoricalParam(choices=list(spec["choices"]))
    if kind == "float":
        return FloatParam(
            distribution=spec["distribution"], mu=spec.get("mu"),
            sigma=spec.get("sigma"), low=spec.get("low"),
            high=spec.get("high"), log_scale=bool(spec.get("log_scale", False)),
        )
    if kind == "int":
        return IntParam(
            distribution=spec["distribution"], mu=spec.get("mu"),
            sigma=spec.get("sigma"), q=spec.get("q", 1),
            low=spec.get("low"), high=spec.get("high"),
        )
    raise ContractError(f"unsupported policy parameter kind {kind!r}")


def _runtime_space(space: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _param_from_spec(spec) for name, spec in space.items()}


def _policy_optimizer(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = manifest["calibration_policy"]
    if set(policy) != {"name", "version", "policy_ref", "options"}:
        raise ContractError("calibration_policy must contain the exact sealed fields")
    contract.validate_artifact_ref(policy["policy_ref"], "calibration_policy.policy_ref")
    options = policy["options"]
    if not isinstance(options, Mapping) or set(options) != {"device", "optimizer"}:
        raise ContractError("calibration policy options must be exactly device and optimizer")
    if options["device"] != manifest["runtime"]["device"]:
        raise ContractError("calibration policy device differs from runtime device")
    optimizer = options["optimizer"]
    required = {"backend", "direction", "seed", "trials_per_strategy", "method_space"}
    if not isinstance(optimizer, Mapping) or set(optimizer) != required:
        raise ContractError("calibration optimizer keys differ from the sealed contract")
    if optimizer["backend"] != "optuna" or optimizer["direction"] != "maximize":
        raise ContractError("calibration requires maximizing fresh Optuna studies")
    if type(optimizer["seed"]) is not int or optimizer["seed"] < 0:
        raise ContractError("optimizer seed must be a non-negative integer")
    if type(optimizer["trials_per_strategy"]) is not int or optimizer["trials_per_strategy"] < 1:
        raise ContractError("trials_per_strategy must be positive")
    if not isinstance(optimizer["method_space"], Mapping) or not optimizer["method_space"]:
        raise ContractError("optimizer method_space must be a non-empty object")
    return optimizer


def _validate_policy_seal(store: Any, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    policy_ref = manifest["calibration_policy"]["policy_ref"]
    policy = _read_ref(store, policy_ref, "desired-results policy")
    policy_module = _load_sibling("desired_results_policy")
    expected = policy_module.calibration_policy_for(
        policy, policy_ref, manifest["target"]["target_id"], manifest["method"],
    )
    if expected != manifest["calibration_policy"]:
        raise ContractError("calibration manifest policy controls differ from sealed policy")
    return policy


def _study_seed(base_seed: int, manifest_sha256: str, method: str, strategy: str) -> int:
    material = contract.canonical_json({
        "base_seed": base_seed, "manifest_sha256": manifest_sha256,
        "method": method, "strategy": strategy,
    })
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _effective_space(method: str, hidden_size: int, layers: Sequence[int], strategy: str,
                     optimizer_policy: Mapping[str, Any]) -> dict[str, Any]:
    if method not in METHODS or strategy not in STRATEGIES:
        raise ContractError("method/strategy is outside the exact calibration matrix")
    if type(hidden_size) is not int or hidden_size < 1:
        raise ContractError("model hidden_size must be positive")
    unique_layers = sorted(set(layers))
    if (not unique_layers or unique_layers != list(layers) or
            unique_layers != list(range(1, max(unique_layers) + 1))):
        raise ContractError("activation layers must be the exact sorted contiguous 1-based range")
    policy_module = _load_sibling("desired_results_policy")
    effective = policy_module.get_effective_method_space(
        method, max(unique_layers), hidden_size,
    )
    declared = optimizer_policy["method_space"]
    if declared != effective:
        raise ContractError(
            f"effective {method} search space differs from frozen optimizer control"
        )
    parameters = json.loads(contract.canonical_json(effective))
    parameters["extraction_strategy"] = {
        "kind": "categorical", "choices": [strategy],
    }
    return _runtime_space(parameters)


def _normalize_sample(method: str, sample: Mapping[str, Any], hidden_size: int,
                      layer_count: int | None = None) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise ContractError("optimizer sample must be an object")
    normalized = dict(sample)
    for name in sorted(_DIMENSION_KEYS & set(normalized)):
        value = normalized[name]
        if type(value) is not int:
            raise ContractError(f"{method}.{name} must be an integer")
        normalized[name] = max(1, min(value, hidden_size))
    # Conditions are normalized in a fixed, audited order.  This ordering is
    # part of result provenance and prevents dictionary/hash iteration drift.
    for low_name, high_name, strict in _ORDERED_CONDITIONS.get(method, ()):
        if low_name not in normalized or high_name not in normalized:
            continue
        low, high = normalized[low_name], normalized[high_name]
        if low > high:
            normalized[low_name], normalized[high_name] = high, low
            low, high = high, low
        if strict and low == high:
            if isinstance(high, int):
                normalized[high_name] = high + 1
            else:
                normalized[high_name] = math.nextafter(float(high), math.inf)
    if method in {"tetno", "grom"}:
        if type(layer_count) is not int or layer_count < 2:
            raise ContractError(f"{method} requires at least two activation layers")
        sensor = normalized.get("sensor_layer")
        start = normalized.get("steering_start")
        end = normalized.get("steering_end")
        if any(type(value) is not int for value in (sensor, start, end)):
            raise ContractError(f"{method} sample lacks integer sensor/steering layers")
        start, end = sorted((max(1, min(start, layer_count)),
                             max(1, min(end, layer_count))))
        sensor = max(1, min(sensor, layer_count))
        if sensor >= start:
            if start > 1:
                sensor = start - 1
            else:
                sensor = 1
                start = 2
                end = max(end, start)
        normalized.update({"sensor_layer": sensor,
                           "steering_start": start,
                           "steering_end": end})
    return normalized


def _finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{label} must be a finite numeric score")
    return float(value)


def _pair_text_map(document: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    rows = document.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ContractError("pair-text artifact must contain a non-empty pairs list")
    result: dict[int, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ContractError("pair-text row must be an object")
        pair_id = row.get("pair_id", index)
        positive = row.get("positive", row.get("positive_response"))
        negative = row.get("negative", row.get("negative_response"))
        if isinstance(positive, Mapping):
            positive = positive.get("model_response")
        if isinstance(negative, Mapping):
            negative = negative.get("model_response")
        prompt = row.get("prompt")
        if type(pair_id) is not int or not all(isinstance(x, str) for x in (prompt, positive, negative)):
            raise ContractError(f"invalid pair-text identity/content at row {index}")
        if pair_id in result:
            raise ContractError("pair-text artifact contains duplicate pair_id")
        result[pair_id] = {"prompt": prompt, "positive": positive, "negative": negative}
    return result


def _write_pair_file(path: Path, benchmark: str, rows: Sequence[Mapping[str, Any]],
                     texts: Mapping[int, Mapping[str, str]]) -> None:
    pairs = []
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id not in texts:
            raise ContractError(f"pair-text artifact misses pair_id {pair_id}")
        text = texts[pair_id]
        pairs.append({
            "pair_id": pair_id, "prompt": text["prompt"],
            "positive_response": {"model_response": text["positive"]},
            "negative_response": {"model_response": text["negative"]},
        })
    payload = {"task_name": benchmark, "num_pairs": len(pairs),
               "pair_ids": [row["pair_id"] for row in rows], "pairs": pairs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json(payload))


def _exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ContractError(f"{label} keys differ from the sealed preflight schema: {actual}")
    return value


def _materialize_activation(
    artifact: Mapping[str, Any], root: Path,
    cache: dict[tuple[str, str, str, str, str, int], Path],
) -> Path:
    identity = (
        artifact["repo_id"], artifact["repo_type"], artifact["revision"],
        artifact["path"], artifact["lfs_sha256"], artifact["size"],
    )
    existing = cache.get(identity)
    if existing is not None:
        return existing
    try:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(
            repo_id=artifact["repo_id"], repo_type=artifact["repo_type"],
            revision=artifact["revision"], filename=artifact["path"],
        ))
    except Exception as exc:
        raise ContractError(f"cannot download exact sealed activation artifact: {exc}") from exc
    if not source.is_file():
        raise ContractError("exact sealed activation download did not return a file")

    activation_root = root / "sealed-activations"
    activation_root.mkdir(parents=True, exist_ok=True)
    destination = activation_root / f"{artifact['lfs_sha256']}.safetensors"
    temporary = activation_root / f".{artifact['lfs_sha256']}.{len(cache)}.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        try:
            with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    size += len(chunk)
                    if size > artifact["size"]:
                        raise ContractError("sealed activation exceeds its proven byte size")
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except OSError as exc:
            raise ContractError(f"cannot materialize exact sealed activation artifact: {exc}") from exc
        if size != artifact["size"] or digest.hexdigest() != artifact["lfs_sha256"]:
            raise ContractError("sealed activation bytes differ from proven LFS SHA/size")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing_digest = hashlib.sha256()
            existing_size = 0
            try:
                with destination.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        existing_size += len(chunk)
                        existing_digest.update(chunk)
            except OSError as exc:
                raise ContractError(f"cannot verify local sealed activation cache: {exc}") from exc
            if existing_size != size or existing_digest.hexdigest() != digest.hexdigest():
                raise ContractError("local sealed activation cache has conflicting bytes")
        cache[identity] = destination
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_activation_route(
    store: Any, route_value: Mapping[str, Any], manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any], root: Path,
    cache: dict[tuple[str, str, str, str, str, int], Path], index: int,
) -> tuple[str, int, Path, Mapping[str, Any]]:
    route = _exact_keys(
        route_value, {"strategy", "layer", "completion_ref", "proof_ref"},
        f"activation route {index}",
    )
    strategy, layer = route["strategy"], route["layer"]
    if strategy not in STRATEGIES or type(layer) is not int or layer < 1:
        raise ContractError(f"activation route {index} identity is invalid")
    completion_ref = contract.validate_artifact_ref(
        route["completion_ref"], f"activation route {index} completion_ref"
    )
    proof_ref = contract.validate_artifact_ref(
        route["proof_ref"], f"activation route {index} proof_ref"
    )
    completion = _exact_keys(_read_ref(
        store, completion_ref, f"completion {strategy}/{layer}"
    ), {
        "schema_version", "complete", "target_id", "route", "proof_ref",
        "activation_lfs_sha256", "activation_header_sha256",
    }, f"completion {strategy}/{layer}")
    proof = _exact_keys(_read_ref(
        store, proof_ref, f"proof {strategy}/{layer}"
    ), {
        "schema_version", "proof_kind", "target_id", "activation_artifact", "route",
        "pair_ids", "tensor_shapes", "tensor_dtypes", "safetensors_header_length",
        "safetensors_header_sha256", "tensor_payload_downloaded",
    }, f"proof {strategy}/{layer}")
    route_identity = {"strategy": strategy, "layer": layer}
    target_id = manifest["target"]["target_id"]
    if (completion["schema_version"] != 2 or completion["complete"] is not True or
            proof["schema_version"] != 2 or
            proof["proof_kind"] != "pinned_hf_safetensors_header" or
            completion["target_id"] != target_id or proof["target_id"] != target_id or
            completion["route"] != route_identity or proof["route"] != route_identity or
            proof["tensor_payload_downloaded"] is not False):
        raise ContractError("sealed activation completion/proof route identity differs")
    if contract.validate_artifact_ref(
            completion["proof_ref"], f"completion {strategy}/{layer}.proof_ref") != proof_ref:
        raise ContractError("sealed activation completion binds a different proof ref")

    artifact = _exact_keys(proof["activation_artifact"], {
        "repo_id", "repo_type", "revision", "path", "lfs_sha256", "size",
    }, f"proof {strategy}/{layer}.activation_artifact")
    expected_path = (
        f"activations/{manifest['target']['model_slug']}/"
        f"{manifest['target']['benchmark']}/{strategy}/layer_{layer}.safetensors"
    )
    expected_revision = manifest["revisions"]["activation"]
    target_revision = target_manifest["revisions"]["activation_revision"]
    sha = artifact["lfs_sha256"]
    header_sha = proof["safetensors_header_sha256"]
    header_length = proof["safetensors_header_length"]
    if (artifact["repo_type"] not in {"dataset", "model", "space"} or
            not isinstance(artifact["repo_id"], str) or not artifact["repo_id"] or
            artifact["revision"] != expected_revision or artifact["revision"] != target_revision or
            artifact["path"] != expected_path or
            not isinstance(sha, str) or len(sha) != 64 or
            any(character not in "0123456789abcdef" for character in sha) or
            type(artifact["size"]) is not int or artifact["size"] <= 0 or
            type(header_length) is not int or header_length <= 1 or
            8 + header_length > artifact["size"] or
            not isinstance(header_sha, str) or len(header_sha) != 64 or
            any(character not in "0123456789abcdef" for character in header_sha)):
        raise ContractError("sealed activation revision/path/LFS/header identity differs")
    if (completion["activation_lfs_sha256"] != sha or
            completion["activation_header_sha256"] != header_sha):
        raise ContractError("sealed activation completion and proof hashes differ")
    expected_pair_ids = list(range(target_manifest["target"]["expected_pairs"]))
    if proof["pair_ids"] != expected_pair_ids:
        raise ContractError("sealed activation proof does not cover exact canonical pair support")

    activation_path = _materialize_activation(artifact, root, cache)
    try:
        with activation_path.open("rb") as handle:
            prefix = handle.read(8)
            observed_header_length = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else None
            header = handle.read(header_length)
    except (OSError, struct.error) as exc:
        raise ContractError(f"cannot inspect sealed activation header: {exc}") from exc
    if (observed_header_length != header_length or len(header) != header_length or
            hashlib.sha256(header).hexdigest() != header_sha):
        raise ContractError("sealed activation header differs from its proof")
    return strategy, layer, activation_path, proof


def _materialize_inputs(store: Any, manifest: Mapping[str, Any], target_manifest: Mapping[str, Any],
                        root: Path) -> tuple[dict[tuple[str, int], str], str, dict[str, Any]]:
    support = manifest["support"]
    train_ids = [row["pair_id"] for row in support["train"]]
    validation_ids = [row["pair_id"] for row in support["validation"]]

    # Resolve and seal every activation before reading calibration text or loading
    # model/tokenizer state.  A bad late route therefore cannot leak any tokens.
    activation_cache: dict[tuple[str, str, str, str, str, int], Path] = {}
    prepared: list[tuple[str, int, Path, Mapping[str, Any]]] = []
    for index, route in enumerate(manifest["activation_routes"]):
        prepared.append(_prepare_activation_route(
            store, route, manifest, target_manifest, root, activation_cache, index,
        ))

    pair_ref = target_manifest["support"].get("pair_texts_ref")
    if pair_ref is None:
        raise ContractError("target manifest does not bind pair_texts_ref")
    pair_document = _read_ref(store, pair_ref, "pair texts")
    texts = _pair_text_map(pair_document)
    validation_file = root / "validation_pairs.json"
    _write_pair_file(validation_file, manifest["target"]["benchmark"], support["validation"], texts)
    train_rows = []
    for row in support["train"]:
        pair_id = row["pair_id"]
        if pair_id not in texts:
            raise ContractError(f"pair-text artifact misses pair_id {pair_id}")
        text = texts[pair_id]
        train_rows.append({
            "pair_id": pair_id, "prompt": text["prompt"],
            "positive_response": {"model_response": text["positive"]},
            "negative_response": {"model_response": text["negative"]},
        })

    from wisent.core.reading.modules.utilities.data.enriched_builder import build_enriched_from_local_strict
    outputs: dict[tuple[str, int], str] = {}
    for strategy, layer, activation_path, proof in prepared:
        route_dir = root / "strict_train" / strategy / f"layer_{layer}"
        route_dir.mkdir(parents=True)
        built_path = Path(build_enriched_from_local_strict(
            manifest["target"]["model_name"], manifest["target"]["benchmark"],
            layer, strategy, str(route_dir), train_ids,
            activation_file=str(activation_path), activation_pair_ids=proof["pair_ids"],
            pair_rows=train_rows,
        ))
        enriched = json.loads(built_path.read_bytes())
        if enriched.get("pair_ids") != train_ids or len(enriched.get("pairs", [])) != len(train_ids):
            raise ContractError("strict enriched input changed exact train support/order")
        outputs[(strategy, layer)] = str(built_path)
    expected = {(strategy, layer) for strategy in STRATEGIES for layer in sorted({r["layer"] for r in manifest["activation_routes"]})}
    if set(outputs) != expected:
        raise ContractError("activation routes do not form the exact seven-strategy layer matrix")
    return outputs, str(validation_file), pair_document


def _load_model(manifest: Mapping[str, Any]):
    from wisent.core.primitives.models import WisentModel
    model = WisentModel(
        manifest["target"]["model_name"], device=manifest["runtime"]["device"],
        revision=manifest["revisions"]["model"],
        tokenizer_revision=manifest["revisions"]["tokenizer"],
    )
    if model.resolved_model_revision != manifest["revisions"]["model"]:
        raise ContractError("loaded model revision differs from immutable manifest pin")
    if model.resolved_tokenizer_revision != manifest["revisions"]["tokenizer"]:
        raise ContractError("loaded tokenizer revision differs from immutable manifest pin")
    return model


def _selected_config(method: str, best: Mapping[str, Any]) -> dict[str, Any]:
    raw_params = best.get("best_params")
    strategy = best.get("strategy")
    if not isinstance(raw_params, Mapping):
        raise ContractError("selected optimizer result lacks normalized best_params")
    params = dict(raw_params)
    if strategy not in STRATEGIES or params.get("extraction_strategy") != strategy:
        raise ContractError("selected optimizer strategy differs from best_params")
    selected: dict[str, Any] = {
        "method": method,
        "strategy": strategy,
        "params": params,
    }
    if method in {"tetno", "grom"}:
        sensor = params.get("sensor_layer")
        start, end = params.get("steering_start"), params.get("steering_end")
        if (any(type(value) is not int or value < 1 for value in (sensor, start, end))
                or not sensor < start <= end):
            raise ContractError(f"{method} selected result lacks valid sensor/steering layer identity")
        selected.update({
            "sensor_layer": sensor,
            "steering_layers": list(range(start, end + 1)),
        })
    else:
        layer = params.get("layer")
        if type(layer) is not int or layer < 1:
            raise ContractError(f"{method} selected result lacks exact scalar layer identity")
        selected["layer"] = layer
    contract.validate_selected_config(selected)
    return selected


def _execute_optimizer(manifest: Mapping[str, Any], optimizer_policy: Mapping[str, Any],
                       cached_model: Any, enriched: Mapping[tuple[str, int], str],
                       validation_file: str, work_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import create_objective
    from wisent.core.utils.services.optimization.core.atoms import BaseOptimizer, HPOConfig
    method = manifest["method"]
    layers = sorted({route["layer"] for route in manifest["activation_routes"]})
    hidden_size = int(cached_model.hidden_size)
    trials = optimizer_policy["trials_per_strategy"]
    base_seed = optimizer_policy["seed"]
    format_results: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        strategy_root = work_root / strategy
        strategy_root.mkdir(parents=True)
        real_objective = create_objective(
            method=method, model=manifest["target"]["model_name"],
            task=manifest["target"]["benchmark"], num_layers=max(layers) + 1,
            limit=None, device=manifest["runtime"]["device"], work_dir=str(strategy_root),
            test_pairs_file=validation_file, strict_enriched_files=dict(enriched),
            cached_model=cached_model,
        )
        space = _effective_space(method, hidden_size, layers, strategy, optimizer_policy)
        canonical_space = _serialize_space(space)
        def objective(raw: Mapping[str, Any], _real=real_objective) -> float:
            normalized = _normalize_sample(method, raw, hidden_size, max(layers))
            return _finite_score(_real(normalized), f"{strategy} objective")
        seed_u64 = _study_seed(base_seed, manifest["manifest_sha256"], method, strategy)
        optuna_seed = seed_u64 % (2 ** 32)
        optimizer = BaseOptimizer()
        optimizer.direction = "maximize"
        result = optimizer.optimize_fn(
            objective, space, trials,
            cfg=HPOConfig(
                backend="optuna", n_trials=trials, seed=optuna_seed,
                sampler="random", pruner="nop", storage=None,
                study_name=None, load_if_exists=False,
            ),
            extra_trials=0,
        )
        if result.backend != "optuna" or result.n_trials != trials or len(result.all_trials) != trials:
            raise ContractError(f"{strategy} did not complete exactly {trials} fresh Optuna trials")
        observed = []
        for trial in result.all_trials:
            if not isinstance(trial, Mapping) or set(trial) != {"params", "score"}:
                raise ContractError(f"{strategy} returned a malformed Optuna trial")
            params = _normalize_sample(method, trial["params"], hidden_size, max(layers))
            score = _finite_score(trial["score"], f"{strategy} trial score")
            observed.append({"params": params, "score": score})
        best_params = _normalize_sample(method, result.best_params, hidden_size, max(layers))
        best_score = _finite_score(result.best_score, f"{strategy} best score")
        if not any(row["params"] == best_params and row["score"] == best_score for row in observed):
            raise ContractError(f"{strategy} best result is not an observed trial")
        format_results.append({
            "strategy": strategy, "seed_u64": seed_u64, "optuna_seed": optuna_seed,
            "trial_count": trials, "effective_space": canonical_space,
            "condition_order": [list(item) for item in _ORDERED_CONDITIONS.get(method, ())],
            "trials": observed, "best_params": best_params,
            "best_validation_score": best_score,
        })
    best = max(format_results, key=lambda row: (row["best_validation_score"],
                                                contract.canonical_sha256(row["best_params"])))
    return format_results, _selected_config(method, best)


def _run_calibration(store: Any, manifest_uri: str, manifest_generation: str,
                     attempt_number: int, *, runtime_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest, manifest_ref, target_manifest = _load_manifest(store, manifest_uri, manifest_generation)
    contract.calibration_attempt_id(manifest["manifest_sha256"], attempt_number)
    if manifest["method"] not in METHODS:
        raise ContractError("calibration manifest method is outside the exact eight-method matrix")
    optimizer_policy = _policy_optimizer(manifest)
    observed_runtime = (
        contract.observe_runtime_evidence()
        if runtime_evidence is None else contract.validate_runtime_evidence(runtime_evidence)
    )
    if (observed_runtime["runtime_revision"] != manifest["revisions"]["runtime"] or
            observed_runtime["runtime_revision"] != manifest["revisions"]["code"] or
            observed_runtime["device"] != manifest["runtime"]["device"] or
            observed_runtime["device"] != manifest["calibration_policy"]["options"]["device"]):
        raise ContractError("observed detached revision/device differs from calibration manifest policy")
    _validate_policy_seal(store, manifest)

    work_root = Path(tempfile.mkdtemp(prefix="desired-results-calibration-v3-"))
    try:
        enriched, validation_file, pair_document = _materialize_inputs(
            store, manifest, target_manifest, work_root
        )
        cached_model = _load_model(manifest)
        format_results, selected_config = _execute_optimizer(manifest, optimizer_policy, cached_model, enriched, validation_file,
        work_root / "trials",)
        selected_config = dict(contract.validate_selected_config(selected_config, "optimizer selected_config"))
        if selected_config["method"] != manifest["method"]:
            raise ContractError("optimizer selected config method differs from calibration manifest")
        expected_params = set(optimizer_policy["method_space"]) | {"extraction_strategy"}
        if set(selected_config["params"]) != expected_params:
            raise ContractError("optimizer selected config does not preserve the full effective parameter set")
        if not set(contract.selected_config_route_keys(selected_config)).issubset(enriched):
            raise ContractError("optimizer selected config requires an undeclared activation route")
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "manifest_ref": manifest_ref,
            "manifest_sha256": manifest["manifest_sha256"],
            "attempt": attempt_number,
            "attempt_id": contract.calibration_attempt_id(manifest["manifest_sha256"], attempt_number),
            "target": manifest["target"], "method": manifest["method"],
            "revisions": manifest["revisions"],
            "fit_support": manifest["support"]["train"],
            "selection_support": manifest["support"]["validation"],
            "test_reads": 0, "test_pair_ids_read": [],
            "strategies": list(STRATEGIES),
            "trials_per_strategy": optimizer_policy["trials_per_strategy"],
            "trial_count": len(STRATEGIES) * optimizer_policy["trials_per_strategy"],
            "optimizer_policy": dict(optimizer_policy),
            "per_strategy": format_results,
            "selected_config": selected_config,
            "runtime_evidence": observed_runtime,
            "runtime_evidence_sha256": contract.runtime_evidence_sha256(observed_runtime),
            "input_evidence": {
                "target_manifest_ref": manifest["target_manifest_ref"],
                "pair_texts_sha256": contract.canonical_sha256(pair_document),
                "activation_route_count": len(enriched),
            },
        }
        uri = (
            f"{manifest['output_namespace'].rstrip('/')}/calibration-v3/"
            f"{manifest['manifest_sha256']}/attempt-{attempt_number}/result.json"
        )
        result_ref = _create_once(store, uri, result)
        return {"selected_config": selected_config, "result_ref": result_ref,
                "runtime_evidence": observed_runtime, "result": result}
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
