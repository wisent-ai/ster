#!/usr/bin/env python3
"""Immutable schema-v3 contracts for desired-results calibration and final execution.

This module deliberately contains no storage or orchestration code.  It defines the
canonical JSON representation and the fail-closed schemas shared by planners,
calibration workers, final-test workers, and finalizers.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.steering import desired_results_target
except (ImportError, ModuleNotFoundError):
    _TARGET_PATH = Path(__file__).with_name("desired_results_target.py")
    _TARGET_SPEC = importlib.util.spec_from_file_location("desired_results_target_v2", _TARGET_PATH)
    if _TARGET_SPEC is None or _TARGET_SPEC.loader is None:
        raise ImportError(f"cannot load target contract from {_TARGET_PATH}")
    desired_results_target = importlib.util.module_from_spec(_TARGET_SPEC)
    _TARGET_SPEC.loader.exec_module(desired_results_target)

SCHEMA_VERSION = 3
ARTIFACT_REF_KEYS = frozenset({"uri", "generation", "size", "sha256"})
RUNTIME_EVIDENCE_KEYS = frozenset({"runtime_revision", "device", "packages"})
RUNTIME_PACKAGES = ("python", "torch", "transformers", "accelerate", "safetensors", "optuna", "wisent")
OPTIONAL_RUNTIME_DISTRIBUTIONS = RUNTIME_PACKAGES[1:]
CALIBRATION_STATES = ("claim", "prepared", "running", "success", "failure")
ATTEMPT_PHASES = (
    "claimed", "inputs_verified", "fit_complete", "test_token_consumed",
    "evaluated", "artifacts_published", "completed",
)
MAX_CALIBRATION_ATTEMPTS = 3
MAX_PRE_TEST_ATTEMPTS = 3
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_POSITIVE_DECIMAL_RE = re.compile(r"[1-9][0-9]*\Z")
_ID_COMPONENT_RE = re.compile(r"[A-Za-z0-9._-]+\Z")


class ContractError(ValueError):
    """A schema-v3 document is malformed, inconsistent, or not content-bound."""


def _reject_non_json(value: Any, path: str = "value") -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} contains a non-string object key")
            _reject_non_json(item, f"{path}.{key}")
        return
    raise ContractError(f"{path} contains non-JSON value {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Encode the sole canonical JSON form used by every v3 identity."""
    _reject_non_json(value)
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContractError(f"value is not canonical-JSON encodable: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _reject_forbidden_keys(value: Any, forbidden: set[str], label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.lower() in forbidden:
                raise ContractError(f"{label} cannot contain held-out field {key!r}")
            _reject_forbidden_keys(item, forbidden, label)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item, forbidden, label)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be boolean")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact(value: Any, keys: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ContractError(f"{label} keys must be exactly {sorted(keys)!r}; got {actual!r}")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _string(value, label)
    if _ID_COMPONENT_RE.fullmatch(value) is None or value in {".", ".."}:
        raise ContractError(f"{label} must be a path-safe identity component")
    return value


def _strings(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"{label} must be a list of non-empty strings")
    if nonempty and not value:
        raise ContractError(f"{label} cannot be empty")
    if len(value) != len(set(value)):
        raise ContractError(f"{label} must not contain duplicates")
    return value


def _finalize(payload_without_derived: Mapping[str, Any], *, id_field: str, hash_field: str,
              kind: str, validator: Any) -> dict[str, Any]:
    if not isinstance(payload_without_derived, Mapping):
        raise ContractError(f"{kind} payload must be an object")
    payload = dict(payload_without_derived)
    forbidden = {id_field, hash_field} & set(payload)
    if forbidden:
        raise ContractError(f"{kind} finalizer derives {sorted(forbidden)!r}")
    digest = canonical_sha256(payload)
    payload[id_field] = content_id(kind, digest)
    payload[hash_field] = canonical_sha256(payload)
    validator(payload)
    return payload


def _validate_content(document: Mapping[str, Any], *, id_field: str, hash_field: str,
                      kind: str) -> None:
    digest = _sha(document[hash_field], hash_field)
    unhashed = dict(document)
    del unhashed[hash_field]
    if digest != canonical_sha256(unhashed):
        raise ContractError(f"{hash_field} does not match canonical payload")
    identity_payload = dict(unhashed)
    identity_payload.pop(id_field)
    expected = content_id(kind, canonical_sha256(identity_payload))
    if document[id_field] != expected:
        raise ContractError(f"{id_field} does not match canonical content identity")


def content_id(kind: str, digest_or_payload: Any) -> str:
    """Return ``kind:<sha256>`` for a digest or canonical-JSON payload."""
    kind = _identifier(kind, "kind")
    digest = digest_or_payload if isinstance(digest_or_payload, str) and _SHA_RE.fullmatch(digest_or_payload) else canonical_sha256(digest_or_payload)
    return f"{kind}:{digest}"


def validate_artifact_ref(value: Any, label: str = "artifact_ref") -> dict[str, str]:
    """Validate an immutable, generation-pinned, byte-exact object reference."""
    ref = _exact(value, ARTIFACT_REF_KEYS, label)
    uri = _string(ref["uri"], f"{label}.uri")
    generation = _string(ref["generation"], f"{label}.generation")
    size = ref["size"]
    if not isinstance(size, str) or _POSITIVE_DECIMAL_RE.fullmatch(size) is None:
        raise ContractError(f"{label}.size must be a canonical positive decimal string")
    _sha(ref["sha256"], f"{label}.sha256")
    return {"uri": uri, "generation": generation, "size": size, "sha256": ref["sha256"]}


def artifact_ref(uri: str, generation: str, size: str, sha256: str) -> dict[str, str]:
    """Construct and validate an ArtifactRef without coercing object identity."""
    return validate_artifact_ref({"uri": uri, "generation": generation, "size": size, "sha256": sha256})


def validate_artifact_binding(value: Any, document: Any, label: str = "artifact_ref") -> dict[str, str]:
    """Bind an ArtifactRef to the exact canonical bytes of an available document."""
    ref = validate_artifact_ref(value, label)
    encoded = canonical_json(document)
    if ref["size"] != str(len(encoded)) or ref["sha256"] != hashlib.sha256(encoded).hexdigest():
        raise ContractError(f"{label} does not match the exact referenced canonical bytes")
    return ref


def _same_ref(left: Any, right: Any, label: str) -> None:
    if validate_artifact_ref(left, f"{label}.left") != validate_artifact_ref(right, f"{label}.right"):
        raise ContractError(f"{label} ArtifactRefs differ")


def validate_runtime_evidence(value: Any, label: str = "runtime_evidence") -> dict[str, Any]:
    evidence = _exact(value, RUNTIME_EVIDENCE_KEYS, label)
    runtime_revision = _string(evidence["runtime_revision"], f"{label}.runtime_revision")
    if _REVISION_RE.fullmatch(runtime_revision) is None:
        raise ContractError(f"{label}.runtime_revision must be a lowercase 40-character detached revision")
    device = _string(evidence["device"], f"{label}.device")
    packages = evidence["packages"]
    if not isinstance(packages, Mapping):
        raise ContractError(f"{label}.packages must be an object")
    allowed = set(RUNTIME_PACKAGES)
    if "python" not in packages or not set(packages) <= allowed:
        raise ContractError(f"{label}.packages must contain python and only {sorted(allowed)!r}")
    normalized: dict[str, str] = {}
    for package, version in packages.items():
        normalized[package] = _string(version, f"{label}.packages.{package}")
    return {"runtime_revision": runtime_revision, "device": device, "packages": normalized}


def runtime_evidence_sha256(value: Any) -> str:
    return canonical_sha256(validate_runtime_evidence(value))


def _observe_detached_runtime_revision() -> str:
    """Return the checked-out commit only when this worker is on detached HEAD."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=False, capture_output=True, text=True,
        )
        symbolic = subprocess.run(
            ["git", "-C", str(repo_root), "symbolic-ref", "-q", "HEAD"],
            check=False, capture_output=True, text=True,
        )
    except OSError as exc:
        raise ContractError(f"cannot observe worker Git revision: {exc}") from exc
    revision = head.stdout.strip()
    if head.returncode != 0 or _REVISION_RE.fullmatch(revision) is None:
        raise ContractError("cannot observe an exact 40-character worker Git revision")
    if symbolic.returncode != 1 or symbolic.stdout.strip():
        raise ContractError("worker checkout must be at detached HEAD")
    return revision


def _observe_device_backend() -> str:
    """Return the accelerator backend that the installed runtime can actually use."""
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception as exc:
        raise ContractError(f"cannot observe worker device backend: {exc}") from exc
    return "cpu"


def observe_runtime_evidence() -> dict[str, Any]:
    """Observe detached source, usable device backend, and installed package versions."""
    packages = {"python": platform.python_version()}
    for distribution in OPTIONAL_RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    evidence = {
        "runtime_revision": _observe_detached_runtime_revision(),
        "device": _observe_device_backend(),
        "packages": packages,
    }
    return validate_runtime_evidence(evidence)


def _validate_runtime_binding(document: Mapping[str, Any], label: str) -> None:
    evidence = validate_runtime_evidence(document["runtime_evidence"], f"{label}.runtime_evidence")
    digest = _sha(document["runtime_evidence_sha256"], f"{label}.runtime_evidence_sha256")
    if digest != canonical_sha256(evidence):
        raise ContractError(f"{label}.runtime_evidence_sha256 differs from evidence")


def _validate_protocol(value: Any, label: str) -> Mapping[str, Any]:
    protocol = _exact(value, {"id", "revision"}, label)
    _identifier(protocol["id"], f"{label}.id")
    _integer(protocol["revision"], f"{label}.revision", minimum=1)
    return protocol


def _validate_revisions(value: Any, label: str) -> Mapping[str, Any]:
    revisions = _exact(value, {"model", "tokenizer", "activation", "code", "runtime"}, label)
    for name in ("model", "tokenizer", "activation", "code", "runtime"):
        revision = revisions[name]
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ContractError(f"{label}.{name} must be a lowercase 40-character immutable revision")
    if revisions["runtime"] != revisions["code"]:
        raise ContractError(f"{label}.runtime must equal the detached code revision")
    return revisions


def _validate_evaluator(value: Any, label: str) -> Mapping[str, Any]:
    evaluator = _exact(value, {"name", "version", "options"}, label)
    _string(evaluator["name"], f"{label}.name")
    _string(evaluator["version"], f"{label}.version")
    if not isinstance(evaluator["options"], Mapping):
        raise ContractError(f"{label}.options must be an object")
    _reject_non_json(evaluator["options"], f"{label}.options")
    return evaluator


def _validate_calibration_policy(value: Any, label: str) -> Mapping[str, Any]:
    policy = _exact(value, {"name", "version", "policy_ref", "options"}, label)
    _string(policy["name"], f"{label}.name")
    _string(policy["version"], f"{label}.version")
    validate_artifact_ref(policy["policy_ref"], f"{label}.policy_ref")
    options = _exact(policy["options"], {"device", "optimizer"}, f"{label}.options")
    _string(options["device"], f"{label}.options.device")
    optimizer = _exact(
        options["optimizer"],
        {"backend", "direction", "seed", "trials_per_strategy", "method_space"},
        f"{label}.options.optimizer",
    )
    _string(optimizer["backend"], f"{label}.options.optimizer.backend")
    _string(optimizer["direction"], f"{label}.options.optimizer.direction")
    _integer(optimizer["seed"], f"{label}.options.optimizer.seed")
    trials = optimizer["trials_per_strategy"]
    if isinstance(trials, Mapping):
        if not trials or any(not isinstance(method, str) or not method for method in trials):
            raise ContractError(f"{label}.options.optimizer.trials_per_strategy must have non-empty method keys")
        for method, count in trials.items():
            _integer(count, f"{label}.options.optimizer.trials_per_strategy.{method}", minimum=1)
    else:
        _integer(trials, f"{label}.options.optimizer.trials_per_strategy", minimum=1)
    if not isinstance(optimizer["method_space"], Mapping) or not optimizer["method_space"]:
        raise ContractError(f"{label}.options.optimizer.method_space must be a non-empty object")
    _reject_non_json(optimizer["method_space"], f"{label}.options.optimizer.method_space")
    return policy


def _validate_pair_rows(value: Any, label: str, *, allow_empty: bool = False) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ContractError(f"{label} must be {'a' if allow_empty else 'a non-empty'} list")
    pair_ids: set[int] = set()
    stable_ids: set[str] = set()
    for index, row_value in enumerate(value):
        row = _exact(row_value, {"pair_id", "stable_id"}, f"{label}[{index}]")
        pair_id = _integer(row["pair_id"], f"{label}[{index}].pair_id")
        stable_id = _string(row["stable_id"], f"{label}[{index}].stable_id")
        if pair_id in pair_ids or stable_id in stable_ids:
            raise ContractError(f"{label} contains duplicate pair identity")
        pair_ids.add(pair_id)
        stable_ids.add(stable_id)
    return value


CALIBRATION_MANIFEST_KEYS = frozenset({
    "schema_version", "protocol", "target", "target_manifest_ref", "method", "revisions",
    "support", "activation_routes", "calibration_policy", "evaluator", "runtime",
    "output_namespace", "manifest_id", "manifest_sha256",
})


def finalize_calibration_manifest(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="manifest_id", hash_field="manifest_sha256",
                     kind="calibration-manifest-v3", validator=validate_calibration_manifest)


def validate_calibration_manifest(manifest: Mapping[str, Any], target_manifest: Mapping[str, Any] | None = None) -> None:
    root = _exact(manifest, CALIBRATION_MANIFEST_KEYS, "calibration manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("calibration manifest schema_version must be 3")
    _validate_protocol(root["protocol"], "calibration manifest.protocol")
    target = _exact(root["target"], {"target_id", "model_name", "model_slug", "benchmark"}, "calibration manifest.target")
    for key in target:
        _string(target[key], f"calibration manifest.target.{key}")
    validate_artifact_ref(root["target_manifest_ref"], "calibration manifest.target_manifest_ref")
    _identifier(root["method"], "calibration manifest.method")
    revisions = _validate_revisions(root["revisions"], "calibration manifest.revisions")
    support = _exact(root["support"], {"train", "validation"}, "calibration manifest.support")
    train = _validate_pair_rows(support["train"], "calibration manifest.support.train")
    validation = _validate_pair_rows(support["validation"], "calibration manifest.support.validation")
    train_ids = {row["pair_id"] for row in train}; validation_ids = {row["pair_id"] for row in validation}
    train_stable = {row["stable_id"] for row in train}; validation_stable = {row["stable_id"] for row in validation}
    if train_ids & validation_ids or train_stable & validation_stable:
        raise ContractError("calibration train and validation support overlap")
    routes = root["activation_routes"]
    if not isinstance(routes, list) or not routes:
        raise ContractError("calibration manifest.activation_routes must be non-empty")
    route_ids: set[tuple[str, int]] = set()
    for index, route_value in enumerate(routes):
        route = _exact(route_value, {"strategy", "layer", "completion_ref", "proof_ref"}, f"calibration manifest.activation_routes[{index}]")
        strategy = _identifier(route["strategy"], f"calibration manifest.activation_routes[{index}].strategy")
        layer = _integer(route["layer"], f"calibration manifest.activation_routes[{index}].layer", minimum=1)
        if (strategy, layer) in route_ids:
            raise ContractError("calibration manifest contains duplicate activation route")
        if strategy not in desired_results_target.STRATEGIES:
            raise ContractError(f"calibration manifest.activation_routes[{index}].strategy is invalid")
        route_ids.add((strategy, layer))
        validate_artifact_ref(route["completion_ref"], f"calibration manifest.activation_routes[{index}].completion_ref")
        validate_artifact_ref(route["proof_ref"], f"calibration manifest.activation_routes[{index}].proof_ref")
    policy = _validate_calibration_policy(root["calibration_policy"], "calibration manifest.calibration_policy")
    if type(policy["options"]["optimizer"]["trials_per_strategy"]) is not int:
        raise ContractError("calibration manifest trials_per_strategy must be a method-specific positive integer")
    _validate_evaluator(root["evaluator"], "calibration manifest.evaluator")
    runtime = _exact(root["runtime"], {"revision", "device"}, "calibration manifest.runtime")
    if runtime["revision"] != revisions["runtime"]:
        raise ContractError("calibration runtime revision differs from revisions.runtime")
    _string(runtime["device"], "calibration manifest.runtime.device")
    if policy["options"]["device"] != runtime["device"]:
        raise ContractError("calibration policy device differs from manifest runtime device")
    _string(root["output_namespace"], "calibration manifest.output_namespace")
    if target_manifest is not None:
        desired_results_target.validate_target_manifest(target_manifest)
        validate_artifact_binding(root["target_manifest_ref"], target_manifest, "calibration target_manifest_ref")
        source_target = target_manifest["target"]
        expected = {"target_id": source_target["target_id"], "model_name": source_target["model_name"], "model_slug": source_target["model_slug"], "benchmark": source_target["benchmark"]}
        if dict(target) != expected:
            raise ContractError("calibration target identity differs from TargetManifestV2")
        expected_support = target_manifest["support"]["splits"]
        if support["train"] != expected_support["train"] or support["validation"] != expected_support["validation"]:
            raise ContractError("calibration train/validation support differs from TargetManifestV2")
        expected_routes = [{key: route[key] for key in ("strategy", "layer", "completion_ref", "proof_ref")} for route in target_manifest["activation"]["routes"]]
        if routes != expected_routes:
            raise ContractError("calibration activation routes differ from TargetManifestV2")
        if root["method"] not in target_manifest["calibration"]["methods"]:
            raise ContractError("calibration method is not declared by TargetManifestV2")
        target_revisions = target_manifest["revisions"]
        if (revisions["model"] != target_revisions["model_revision"] or
                revisions["tokenizer"] != target_revisions["tokenizer_revision"] or
                revisions["activation"] != target_revisions["activation_revision"]):
            raise ContractError("calibration model/tokenizer/activation revisions differ from TargetManifestV2")
    _validate_content(root, id_field="manifest_id", hash_field="manifest_sha256", kind="calibration-manifest-v3")


def calibration_attempt_id(manifest_sha256: str, attempt: int) -> str:
    _sha(manifest_sha256, "manifest_sha256")
    _integer(attempt, "attempt", minimum=1)
    if attempt > MAX_CALIBRATION_ATTEMPTS:
        raise ContractError(f"attempt must be <= {MAX_CALIBRATION_ATTEMPTS}")
    return content_id("calibration-attempt-v3", {"manifest_sha256": manifest_sha256, "attempt": attempt})


CALIBRATION_RECEIPT_BASE_KEYS = frozenset({
    "schema_version", "manifest_ref", "manifest_sha256", "attempt", "attempt_id", "state",
    "runtime_evidence", "runtime_evidence_sha256", "receipt_id", "receipt_sha256",
})
CALIBRATION_SUCCESS_KEYS = CALIBRATION_RECEIPT_BASE_KEYS | {"selected_config", "result_ref"}
CALIBRATION_FAILURE_KEYS = CALIBRATION_RECEIPT_BASE_KEYS | {"error"}
CALIBRATION_PHASE_KEYS = CALIBRATION_RECEIPT_BASE_KEYS | {"evidence"}


def _finalize_calibration_receipt(payload: Mapping[str, Any], state: str) -> dict[str, Any]:
    if payload.get("state") != state:
        raise ContractError(f"calibration receipt state must be {state!r}")
    return _finalize(payload, id_field="receipt_id", hash_field="receipt_sha256",
                     kind=f"calibration-{state}-receipt-v3", validator=validate_calibration_receipt)


def finalize_calibration_success_receipt(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize_calibration_receipt(payload_without_derived, "success")


def finalize_calibration_failure_receipt(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize_calibration_receipt(payload_without_derived, "failure")


def finalize_calibration_phase_receipt(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    state = payload_without_derived.get("state")
    if state not in {"claim", "prepared", "running"}:
        raise ContractError("phase receipt state must be claim, prepared, or running")
    return _finalize_calibration_receipt(payload_without_derived, state)


_SENSOR_LAYER_METHODS = frozenset({"tetno", "grom"})
_SELECTED_CONFIG_FORBIDDEN_KEYS = {
    "test", "test_support", "test_pair_ids", "test_ids", "held_out", "heldout",
    "held_out_support",
}


def validate_selected_config(value: Mapping[str, Any],
                             label: str = "selected_config") -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    method = _identifier(value.get("method"), f"{label}.method")
    if method not in desired_results_target.METHODS:
        raise ContractError(f"{label}.method is outside the exact method matrix")
    expected = (
        {"method", "strategy", "sensor_layer", "steering_layers", "params"}
        if method in _SENSOR_LAYER_METHODS else
        {"method", "strategy", "layer", "params"}
    )
    selected = _exact(value, expected, label)
    strategy = _identifier(selected["strategy"], f"{label}.strategy")
    if strategy not in desired_results_target.STRATEGIES:
        raise ContractError(f"{label}.strategy is outside the exact strategy matrix")
    params = selected["params"]
    if not isinstance(params, Mapping) or not params:
        raise ContractError(f"{label}.params must preserve all calibrated hyperparameters")
    _reject_non_json(params, f"{label}.params")
    _reject_forbidden_keys(selected, _SELECTED_CONFIG_FORBIDDEN_KEYS, label)
    if params.get("extraction_strategy") != strategy:
        raise ContractError(f"{label}.strategy differs from params.extraction_strategy")
    if method in _SENSOR_LAYER_METHODS:
        sensor = _integer(selected["sensor_layer"], f"{label}.sensor_layer", minimum=1)
        steering = selected["steering_layers"]
        if (not isinstance(steering, list) or not steering or
                any(type(layer) is not int or layer < 1 for layer in steering) or
                steering != list(range(steering[0], steering[-1] + 1))):
            raise ContractError(f"{label}.steering_layers must be an exact contiguous positive layer span")
        if (params.get("sensor_layer") != sensor or
                params.get("steering_start") != steering[0] or
                params.get("steering_end") != steering[-1]):
            raise ContractError(f"{label} sensor/steering identity differs from params")
    else:
        layer = _integer(selected["layer"], f"{label}.layer", minimum=1)
        if params.get("layer") != layer:
            raise ContractError(f"{label}.layer differs from params.layer")
    return selected




def selected_config_route_keys(value: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Return the exact activation routes needed to realize a selected config."""
    selected = validate_selected_config(value)
    strategy = selected["strategy"]
    if selected["method"] in _SENSOR_LAYER_METHODS:
        layers = sorted({selected["sensor_layer"], *selected["steering_layers"]})
    else:
        layers = [selected["layer"]]
    return [(strategy, layer) for layer in layers]


def validate_calibration_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> None:
    state = receipt.get("state") if isinstance(receipt, Mapping) else None
    keys = CALIBRATION_SUCCESS_KEYS if state == "success" else CALIBRATION_FAILURE_KEYS if state == "failure" else CALIBRATION_PHASE_KEYS
    root = _exact(receipt, keys, "calibration receipt")
    if root["schema_version"] != SCHEMA_VERSION or state not in CALIBRATION_STATES:
        raise ContractError("calibration receipt schema/state is invalid")
    ref = validate_artifact_ref(root["manifest_ref"], "calibration receipt.manifest_ref")
    manifest_sha = _sha(root["manifest_sha256"], "calibration receipt.manifest_sha256")
    attempt = _integer(root["attempt"], "calibration receipt.attempt", minimum=1)
    if root["attempt_id"] != calibration_attempt_id(manifest_sha, attempt):
        raise ContractError("calibration receipt attempt_id differs")
    _validate_runtime_binding(root, "calibration receipt")
    if state == "success":
        selected = validate_selected_config(
            root["selected_config"], "calibration receipt.selected_config",
        )
        validate_artifact_ref(root["result_ref"], "calibration receipt.result_ref")
    elif state == "failure":
        error = _exact(root["error"], {"type", "message", "retryable"}, "calibration receipt.error")
        _string(error["type"], "calibration receipt.error.type")
        _string(error["message"], "calibration receipt.error.message")
        _boolean(error["retryable"], "calibration receipt.error.retryable")
        if error["retryable"] and attempt >= MAX_CALIBRATION_ATTEMPTS:
            raise ContractError("last calibration attempt cannot remain retryable")
    else:
        evidence = root["evidence"]
        if not isinstance(evidence, Mapping):
            raise ContractError("calibration phase evidence must be an object")
        _reject_non_json(evidence, "calibration receipt.evidence")
    if manifest is not None:
        validate_calibration_manifest(manifest)
        validate_artifact_binding(ref, manifest, "calibration receipt.manifest_ref")
        if manifest["manifest_sha256"] != manifest_sha:
            raise ContractError("calibration receipt binds a different manifest")
        expected_runtime = manifest["runtime"]
        observed = root["runtime_evidence"]
        if observed["runtime_revision"] != expected_runtime["revision"] or observed["device"] != expected_runtime["device"]:
            raise ContractError("calibration observed runtime differs from manifest")
        if state == "success" and root["selected_config"]["method"] != manifest["method"]:
            raise ContractError("calibration selected config method differs from manifest")
        if state == "success":
            selected = validate_selected_config(
                root["selected_config"], "calibration receipt.selected_config",
            )
            declared = manifest["calibration_policy"]["options"]["optimizer"]["method_space"]
            expected_params = set(declared) | {"extraction_strategy"}
            if set(selected["params"]) != expected_params:
                raise ContractError("calibration selected config does not preserve the full effective parameter set")
            available_routes = {
                (route["strategy"], route["layer"])
                for route in manifest["activation_routes"]
            }
            if not set(selected_config_route_keys(selected)).issubset(available_routes):
                raise ContractError("calibration selected config requires an undeclared activation route")
    _validate_content(root, id_field="receipt_id", hash_field="receipt_sha256", kind=f"calibration-{state}-receipt-v3")


def validate_calibration_success_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> None:
    if receipt.get("state") != "success":
        raise ContractError("expected successful calibration receipt")
    validate_calibration_receipt(receipt, manifest)


def validate_calibration_failure_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> None:
    if receipt.get("state") != "failure":
        raise ContractError("expected failed calibration receipt")
    validate_calibration_receipt(receipt, manifest)


def validate_calibration_transition(previous: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    validate_calibration_receipt(previous); validate_calibration_receipt(candidate)
    immutable = ("manifest_ref", "manifest_sha256", "attempt", "attempt_id", "runtime_evidence", "runtime_evidence_sha256")
    if any(previous[field] != candidate[field] for field in immutable):
        raise ContractError("calibration transition changes immutable attempt identity/evidence")
    allowed = {
        "claim": ("prepared", "failure"),
        "prepared": ("running", "failure"),
        "running": ("success", "failure"),
    }
    expected = allowed.get(previous["state"], ())
    legal = candidate["state"] in expected
    if not legal:
        raise ContractError("calibration receipt transition is not adjacent and monotonic")


def calibration_retry_decision(receipts: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)) or not receipts:
        return "claim"
    for receipt in receipts:
        validate_calibration_receipt(receipt)
    latest = max(receipts, key=lambda item: item["attempt"])
    if latest["state"] == "success":
        return "complete"
    if latest["state"] != "failure":
        return "resume"
    if latest["error"]["retryable"] and latest["attempt"] < MAX_CALIBRATION_ATTEMPTS:
        return "retry"
    return "failed"


EXECUTION_CONTRACT_KEYS = frozenset({
    "schema_version", "protocol", "target_manifest", "target_manifest_ref", "target",
    "revisions", "matrix", "calibration_policy", "calibration_receipts", "evaluator",
    "evaluator_ref", "final_test", "arms", "retry_policy", "output_namespace",
    "runtime_evidence", "runtime_evidence_sha256", "contract_id", "contract_sha256",
})


def finalize_execution_contract(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="contract_id", hash_field="contract_sha256",
                     kind="execution-contract-v3", validator=validate_execution_contract)


def _successful_calibration_set(root: Mapping[str, Any], methods: list[str]) -> dict[str, Mapping[str, Any]]:
    receipts = root["calibration_receipts"]
    if not isinstance(receipts, list) or len(receipts) != len(methods):
        raise ContractError("execution contract requires exactly one calibration receipt per method")
    by_method: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        validate_calibration_success_receipt(receipt)
        result = receipt["selected_config"]
        method = result.get("method") if isinstance(result, Mapping) else None
        if method not in methods or method in by_method:
            raise ContractError("calibration receipt methods must cover the declared methods exactly once")
        by_method[method] = receipt
    if set(by_method) != set(methods):
        raise ContractError("calibration receipt set is incomplete")
    return by_method


def validate_execution_contract(contract: Mapping[str, Any]) -> None:
    root = _exact(contract, EXECUTION_CONTRACT_KEYS, "execution contract")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("execution contract schema_version must be 3")
    _validate_protocol(root["protocol"], "execution contract.protocol")
    manifest = root["target_manifest"]
    try:
        desired_results_target.validate_target_manifest(manifest)
    except desired_results_target.ContractError as exc:
        raise ContractError(f"invalid embedded TargetManifestV2: {exc}") from exc
    validate_artifact_binding(root["target_manifest_ref"], manifest, "execution contract.target_manifest_ref")
    source_target = manifest["target"]
    target = _exact(root["target"], {"target_id", "model_name", "model_slug", "benchmark"}, "execution contract.target")
    expected_target = {key: source_target[key] for key in target}
    if dict(target) != expected_target:
        raise ContractError("execution target dimensions differ from TargetManifestV2")
    revisions = _validate_revisions(root["revisions"], "execution contract.revisions")
    target_revisions = manifest["revisions"]
    if (revisions["model"] != target_revisions["model_revision"] or
            revisions["tokenizer"] != target_revisions["tokenizer_revision"] or
            revisions["activation"] != target_revisions["activation_revision"]):
        raise ContractError("execution model/tokenizer/activation revisions differ from TargetManifestV2")
    matrix = _exact(root["matrix"], {"strategies", "layers", "methods", "pairs", "splits"}, "execution contract.matrix")
    strategies = _strings(matrix["strategies"], "execution contract.matrix.strategies")
    methods = _strings(matrix["methods"], "execution contract.matrix.methods")
    layers = matrix["layers"]
    if not isinstance(layers, list) or any(type(layer) is not int or layer < 1 for layer in layers) or len(layers) != len(set(layers)) or not layers:
        raise ContractError("execution contract.matrix.layers must be unique positive integers")
    _integer(matrix["pairs"], "execution contract.matrix.pairs", minimum=1)
    splits = matrix["splits"]
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation", "test"}:
        raise ContractError("execution contract.matrix.splits must contain train, validation, and test")
    for split, count in splits.items():
        _integer(count, f"execution contract.matrix.splits.{split}")
    if strategies != list(desired_results_target.STRATEGIES) or strategies != manifest["calibration"]["strategies"]:
        raise ContractError("execution strategies differ from the required seven-strategy matrix")
    expected_layers = list(range(1, manifest["calibration"]["layer_count"] + 1))
    if layers != expected_layers or methods != manifest["calibration"]["methods"]:
        raise ContractError("execution layer/method dimensions differ from TargetManifestV2")
    if matrix["pairs"] != source_target["expected_pairs"] or dict(splits) != manifest["support"]["split_counts"]:
        raise ContractError("execution pair/split dimensions differ from TargetManifestV2")
    policy = _validate_calibration_policy(root["calibration_policy"], "execution contract.calibration_policy")
    method_spaces = policy["options"]["optimizer"]["method_space"]
    if set(method_spaces) != set(methods) or any(
        not isinstance(method_spaces[method], Mapping) or not method_spaces[method]
        for method in methods
    ):
        raise ContractError("execution calibration policy must bind one non-empty effective space per method")
    trial_budgets = policy["options"]["optimizer"]["trials_per_strategy"]
    if not isinstance(trial_budgets, Mapping) or set(trial_budgets) != set(methods):
        raise ContractError("execution calibration policy must bind one trials_per_strategy budget per method")
    evaluator = _validate_evaluator(root["evaluator"], "execution contract.evaluator")
    validate_artifact_binding(root["evaluator_ref"], evaluator, "execution contract.evaluator_ref")
    by_method = _successful_calibration_set(root, methods)
    for method, receipt in by_method.items():
        selected = validate_selected_config(
            receipt["selected_config"], f"execution calibration {method}.selected_config",
        )
        expected_params = set(method_spaces[method]) | {"extraction_strategy"}
        if set(selected["params"]) != expected_params:
            raise ContractError(f"execution calibration {method} does not preserve the full parameter set")
        if any(strategy not in strategies or layer not in layers
               for strategy, layer in selected_config_route_keys(selected)):
            raise ContractError(f"execution calibration {method} selects an undeclared activation route")
    evidence = validate_runtime_evidence(root["runtime_evidence"], "execution contract.runtime_evidence")
    _validate_runtime_binding(root, "execution contract")
    if policy["options"]["device"] != evidence["device"]:
        raise ContractError("execution calibration policy device differs from runtime evidence")
    for method, receipt in by_method.items():
        if receipt["runtime_evidence"] != evidence or receipt["runtime_evidence_sha256"] != root["runtime_evidence_sha256"]:
            raise ContractError(f"calibration runtime evidence differs for method {method}")
        if receipt["runtime_evidence"]["runtime_revision"] != revisions["runtime"]:
            raise ContractError("calibration runtime revision differs from execution revisions")
    final_test = _exact(root["final_test"], {"split", "evaluations_per_arm"}, "execution contract.final_test")
    if final_test != {"split": "test", "evaluations_per_arm": 1}:
        raise ContractError("execution final_test must permit exactly one held-out test evaluation per arm")
    arms = _strings(root["arms"], "execution contract.arms")
    if arms != ["baseline", *methods] or len(arms) != len(set(arms)):
        raise ContractError("execution arms must be baseline followed by each distinct method")
    retry = _exact(root["retry_policy"], {"max_pre_test_attempts"}, "execution contract.retry_policy")
    if retry["max_pre_test_attempts"] != MAX_PRE_TEST_ATTEMPTS:
        raise ContractError(f"execution max_pre_test_attempts must be {MAX_PRE_TEST_ATTEMPTS}")
    _string(root["output_namespace"], "execution contract.output_namespace")
    _validate_content(root, id_field="contract_id", hash_field="contract_sha256", kind="execution-contract-v3")


def derive_output_prefix(output_namespace: str, contract_sha256: str, arm: str) -> str:
    namespace = _string(output_namespace, "output_namespace").rstrip("/")
    _sha(contract_sha256, "contract_sha256")
    arm = _identifier(arm, "arm")
    return f"{namespace}/{contract_sha256}/{arm}/"


ARM_MANIFEST_KEYS = frozenset({
    "schema_version", "contract_ref", "contract_sha256", "arm", "method",
    "selected_config_ref", "pair_texts_ref", "support_refs", "evaluator_ref",
    "runtime_evidence", "runtime_evidence_sha256", "output_namespace", "output_prefix",
    "manifest_id", "manifest_sha256",
})


def finalize_arm_manifest(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="manifest_id", hash_field="manifest_sha256",
                     kind="arm-manifest-v3", validator=validate_arm_manifest)


def validate_arm_manifest(manifest: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> None:
    root = _exact(manifest, ARM_MANIFEST_KEYS, "arm manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("arm manifest schema_version must be 3")
    ref = validate_artifact_ref(root["contract_ref"], "arm manifest.contract_ref")
    contract_sha = _sha(root["contract_sha256"], "arm manifest.contract_sha256")
    arm = _identifier(root["arm"], "arm manifest.arm")
    method = root["method"]
    if method is not None:
        _identifier(method, "arm manifest.method")
    selected = root["selected_config_ref"]
    if arm == "baseline":
        if method is not None or selected is not None:
            raise ContractError("baseline arm cannot carry method or selected config")
    else:
        if method != arm or selected is None:
            raise ContractError("method arm must carry matching method and selected config ref")
        validate_artifact_ref(selected, "arm manifest.selected_config_ref")
    pair_texts_ref = validate_artifact_ref(root["pair_texts_ref"], "arm manifest.pair_texts_ref")
    evaluator_ref = validate_artifact_ref(root["evaluator_ref"], "arm manifest.evaluator_ref")
    support_refs = root["support_refs"]
    if not isinstance(support_refs, list):
        raise ContractError("arm manifest.support_refs must be a list")
    support_keys: set[tuple[str, int]] = set()
    for index, route_value in enumerate(support_refs):
        route = _exact(route_value, {"strategy", "layer", "completion_ref", "proof_ref"}, f"arm manifest.support_refs[{index}]")
        strategy = _identifier(route["strategy"], f"arm manifest.support_refs[{index}].strategy")
        layer = _integer(route["layer"], f"arm manifest.support_refs[{index}].layer", minimum=1)
        if (strategy, layer) in support_keys:
            raise ContractError("arm manifest.support_refs contains duplicate strategy/layer")
        support_keys.add((strategy, layer))
        validate_artifact_ref(route["completion_ref"], f"arm manifest.support_refs[{index}].completion_ref")
        validate_artifact_ref(route["proof_ref"], f"arm manifest.support_refs[{index}].proof_ref")
    _validate_runtime_binding(root, "arm manifest")
    namespace = _string(root["output_namespace"], "arm manifest.output_namespace")
    if root["output_prefix"] != derive_output_prefix(namespace, contract_sha, arm):
        raise ContractError("arm manifest.output_prefix is not content-derived")
    if contract is not None:
        validate_execution_contract(contract)
        validate_artifact_binding(ref, contract, "arm manifest.contract_ref")
        _same_ref(pair_texts_ref, contract["target_manifest"]["support"]["pair_texts_ref"], "arm pair texts")
        _same_ref(evaluator_ref, contract["evaluator_ref"], "arm evaluator")
        if contract["contract_sha256"] != contract_sha:
            raise ContractError("arm manifest binds a different execution contract")
        if arm not in contract["arms"]:
            raise ContractError("arm is not declared by execution contract")
        if root["runtime_evidence"] != contract["runtime_evidence"] or root["runtime_evidence_sha256"] != contract["runtime_evidence_sha256"]:
            raise ContractError("arm runtime evidence differs from execution contract")
        if namespace != contract["output_namespace"]:
            raise ContractError("arm output namespace differs from execution contract")
        if arm != "baseline":
            receipt = next(item for item in contract["calibration_receipts"] if item["selected_config"].get("method") == arm)
            _same_ref(selected, receipt["result_ref"], "arm selected config")
            config = receipt["selected_config"]
            expected_keys = set(selected_config_route_keys(config))
            expected_routes = [
                {key: route[key] for key in ("strategy", "layer", "completion_ref", "proof_ref")}
                for route in contract["target_manifest"]["activation"]["routes"]
                if (route["strategy"], route["layer"]) in expected_keys
            ]
            if support_refs != expected_routes or len(expected_routes) != len(expected_keys):
                raise ContractError("arm support refs are not the exact selected method layer routes")
        elif support_refs:
            raise ContractError("baseline arm cannot carry activation support refs")
    _validate_content(root, id_field="manifest_id", hash_field="manifest_sha256", kind="arm-manifest-v3")


def attempt_id(arm_manifest_sha256: str, attempt: int) -> str:
    _sha(arm_manifest_sha256, "arm_manifest_sha256")
    _integer(attempt, "attempt", minimum=1)
    if attempt > MAX_PRE_TEST_ATTEMPTS:
        raise ContractError(f"attempt must be <= {MAX_PRE_TEST_ATTEMPTS}")
    return content_id("final-test-attempt-v3", {"arm_manifest_sha256": arm_manifest_sha256, "attempt": attempt})


def test_token_id(arm_manifest_sha256: str) -> str:
    _sha(arm_manifest_sha256, "arm_manifest_sha256")
    return content_id("final-test-token-v3", {"arm_manifest_sha256": arm_manifest_sha256})


ATTEMPT_RECEIPT_KEYS = frozenset({
    "schema_version", "contract_sha256", "arm_manifest_sha256", "arm", "attempt",
    "attempt_id", "phase", "test_token_id", "evidence", "staged_result_ref",
    "publication_ref", "quarantined", "receipt_id", "receipt_sha256",
})


def finalize_attempt_receipt(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="receipt_id", hash_field="receipt_sha256",
                     kind="attempt-receipt-v3", validator=validate_attempt_receipt)


def validate_attempt_receipt(receipt: Mapping[str, Any], contract: Mapping[str, Any] | None = None,
                             arm_manifest: Mapping[str, Any] | None = None) -> None:
    root = _exact(receipt, ATTEMPT_RECEIPT_KEYS, "attempt receipt")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("attempt receipt schema_version must be 3")
    contract_sha = _sha(root["contract_sha256"], "attempt receipt.contract_sha256")
    arm_sha = _sha(root["arm_manifest_sha256"], "attempt receipt.arm_manifest_sha256")
    _identifier(root["arm"], "attempt receipt.arm")
    attempt = _integer(root["attempt"], "attempt receipt.attempt", minimum=1)
    if root["attempt_id"] != attempt_id(arm_sha, attempt):
        raise ContractError("attempt receipt attempt_id differs")
    phase = root["phase"]
    if phase not in ATTEMPT_PHASES:
        raise ContractError("attempt receipt phase is invalid")
    token = root["test_token_id"]
    phase_index = ATTEMPT_PHASES.index(phase)
    if phase_index < ATTEMPT_PHASES.index("test_token_consumed"):
        if token is not None:
            raise ContractError("pre-test attempt cannot claim a test token")
    elif token != test_token_id(arm_sha):
        raise ContractError("post-test attempt must carry the arm-global test token")
    evidence = root["evidence"]
    if not isinstance(evidence, Mapping):
        raise ContractError("attempt receipt evidence must be an object")
    _reject_non_json(evidence, "attempt receipt.evidence")
    staged = root["staged_result_ref"]; published = root["publication_ref"]
    if staged is not None:
        validate_artifact_ref(staged, "attempt receipt.staged_result_ref")
    if published is not None:
        validate_artifact_ref(published, "attempt receipt.publication_ref")
    quarantined = _boolean(root["quarantined"], "attempt receipt.quarantined")
    if published is not None and staged is None:
        raise ContractError("published attempt requires durable staged output")
    if phase_index < ATTEMPT_PHASES.index("evaluated") and staged is not None:
        raise ContractError("pre-evaluation attempt cannot carry staged output")
    if phase_index < ATTEMPT_PHASES.index("artifacts_published") and published is not None:
        raise ContractError("attempt cannot claim publication before artifacts_published")
    if quarantined and (phase != "completed" or token is None or staged is not None or published is not None):
        raise ContractError("quarantine is terminal, post-token, and has no durable output")
    if phase == "completed" and not quarantined and (staged is None or published is None):
        raise ContractError("successful completed attempt requires staged and published refs")
    if contract is not None:
        validate_execution_contract(contract)
        if contract["contract_sha256"] != contract_sha or root["arm"] not in contract["arms"]:
            raise ContractError("attempt differs from execution contract")
    if arm_manifest is not None:
        validate_arm_manifest(arm_manifest, contract)
        if arm_manifest["manifest_sha256"] != arm_sha or arm_manifest["arm"] != root["arm"]:
            raise ContractError("attempt differs from arm manifest")
    _validate_content(root, id_field="receipt_id", hash_field="receipt_sha256", kind="attempt-receipt-v3")


def validate_attempt_transition(previous: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    validate_attempt_receipt(previous); validate_attempt_receipt(candidate)
    immutable = ("contract_sha256", "arm_manifest_sha256", "arm", "attempt", "attempt_id")
    if any(previous[field] != candidate[field] for field in immutable):
        raise ContractError("attempt transition changes immutable identity")
    previous_index = ATTEMPT_PHASES.index(previous["phase"]); candidate_index = ATTEMPT_PHASES.index(candidate["phase"])
    direct_quarantine = (
        candidate["phase"] == "completed" and candidate["quarantined"]
        and previous_index >= ATTEMPT_PHASES.index("test_token_consumed")
        and previous["staged_result_ref"] is None
    )
    if candidate_index != previous_index + 1 and not direct_quarantine:
        raise ContractError("attempt transition must advance one phase or terminally quarantine a lost post-token result")
    if previous["test_token_id"] is not None and candidate["test_token_id"] != previous["test_token_id"]:
        raise ContractError("attempt transition changes the arm-global test token")
    for field in ("evidence", "staged_result_ref", "publication_ref"):
        old = previous[field]
        if old is not None and candidate[field] != old:
            raise ContractError(f"attempt transition changes immutable {field}")
    if previous["quarantined"]:
        raise ContractError("quarantined attempt is terminal")


def retry_decision(receipts: Sequence[Mapping[str, Any]]) -> str:
    """Return claim/retry/resume_publication/quarantine/complete for one arm."""
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)) or not receipts:
        return "claim"
    for receipt in receipts:
        validate_attempt_receipt(receipt)
    identities = {(receipt["contract_sha256"], receipt["arm_manifest_sha256"], receipt["arm"]) for receipt in receipts}
    if len(identities) != 1:
        raise ContractError("retry receipts span multiple contracts or arms")
    consumed_attempts = {receipt["attempt"] for receipt in receipts if receipt["test_token_id"] is not None}
    if consumed_attempts and max(receipt["attempt"] for receipt in receipts) > min(consumed_attempts):
        raise ContractError("an arm cannot start another attempt after consuming its test token")
    latest = max(receipts, key=lambda item: (item["attempt"], ATTEMPT_PHASES.index(item["phase"])))
    if latest["phase"] == "completed":
        return "quarantine" if latest["quarantined"] else "complete"
    if latest["test_token_id"] is not None:
        return "resume_publication" if latest["staged_result_ref"] is not None else "quarantine"
    return "retry" if latest["attempt"] < MAX_PRE_TEST_ATTEMPTS else "quarantine"


COMPLETION_RECEIPT_KEYS = frozenset({
    "schema_version", "contract_sha256", "arm_manifest_sha256", "arm", "attempt_id",
    "attempt_receipt_ref", "staged_result_ref", "publication_ref", "completion_id",
    "completion_sha256",
})


def finalize_completion_receipt(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="completion_id", hash_field="completion_sha256",
                     kind="completion-receipt-v3", validator=validate_completion_receipt)


def validate_completion_receipt(receipt: Mapping[str, Any], contract: Mapping[str, Any] | None = None,
                                arm_manifest: Mapping[str, Any] | None = None,
                                attempt_receipt: Mapping[str, Any] | None = None) -> None:
    root = _exact(receipt, COMPLETION_RECEIPT_KEYS, "completion receipt")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("completion receipt schema_version must be 3")
    contract_sha = _sha(root["contract_sha256"], "completion receipt.contract_sha256")
    arm_sha = _sha(root["arm_manifest_sha256"], "completion receipt.arm_manifest_sha256")
    _identifier(root["arm"], "completion receipt.arm")
    _string(root["attempt_id"], "completion receipt.attempt_id")
    attempt_ref = validate_artifact_ref(root["attempt_receipt_ref"], "completion receipt.attempt_receipt_ref")
    staged = validate_artifact_ref(root["staged_result_ref"], "completion receipt.staged_result_ref")
    publication = validate_artifact_ref(root["publication_ref"], "completion receipt.publication_ref")
    if contract is not None:
        validate_execution_contract(contract)
        if contract["contract_sha256"] != contract_sha or root["arm"] not in contract["arms"]:
            raise ContractError("completion differs from execution contract")
    if arm_manifest is not None:
        validate_arm_manifest(arm_manifest, contract)
        if arm_manifest["manifest_sha256"] != arm_sha or arm_manifest["arm"] != root["arm"]:
            raise ContractError("completion differs from arm manifest")
    if attempt_receipt is not None:
        validate_attempt_receipt(attempt_receipt, contract, arm_manifest)
        validate_artifact_binding(attempt_ref, attempt_receipt, "completion attempt_receipt_ref")
        if attempt_receipt["phase"] != "completed" or attempt_receipt["quarantined"]:
            raise ContractError("completion requires successful non-quarantined completed attempt")
        for field in ("contract_sha256", "arm_manifest_sha256", "arm", "attempt_id"):
            if root[field] != attempt_receipt[field]:
                raise ContractError(f"completion differs from attempt on {field}")
        if staged != attempt_receipt["staged_result_ref"] or publication != attempt_receipt["publication_ref"]:
            raise ContractError("completion output refs differ from completed attempt")
    _validate_content(root, id_field="completion_id", hash_field="completion_sha256", kind="completion-receipt-v3")


FINAL_SEAL_KEYS = frozenset({
    "schema_version", "contract", "contract_ref", "contract_sha256", "arm_manifest_refs",
    "runtime_evidence", "runtime_evidence_sha256", "seal_id", "seal_sha256",
})


def finalize_final_seal(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="seal_id", hash_field="seal_sha256",
                     kind="final-seal-v3", validator=validate_final_seal)


def validate_final_seal(seal: Mapping[str, Any], arm_manifests: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    root = _exact(seal, FINAL_SEAL_KEYS, "final seal")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("final seal schema_version must be 3")
    contract = root["contract"]
    validate_execution_contract(contract)
    contract_ref = validate_artifact_binding(root["contract_ref"], contract, "final seal.contract_ref")
    contract_sha = _sha(root["contract_sha256"], "final seal.contract_sha256")
    if contract_sha != contract["contract_sha256"]:
        raise ContractError("final seal execution contract identity differs")
    refs = root["arm_manifest_refs"]
    if not isinstance(refs, Mapping) or set(refs) != set(contract["arms"]):
        raise ContractError("final seal arm refs must cover every declared arm exactly once")
    for arm, ref in refs.items():
        validate_artifact_ref(ref, f"final seal.arm_manifest_refs.{arm}")
    _validate_runtime_binding(root, "final seal")
    if root["runtime_evidence"] != contract["runtime_evidence"] or root["runtime_evidence_sha256"] != contract["runtime_evidence_sha256"]:
        raise ContractError("final seal runtime evidence differs from execution contract")
    if arm_manifests is not None:
        if set(arm_manifests) != set(contract["arms"]):
            raise ContractError("provided arm manifests do not cover execution contract")
        for arm, manifest in arm_manifests.items():
            validate_arm_manifest(manifest, contract)
            if manifest["arm"] != arm:
                raise ContractError("final seal arm manifest identity differs")
            validate_artifact_binding(refs[arm], manifest, f"final seal.arm_manifest_refs.{arm}")
    _validate_content(root, id_field="seal_id", hash_field="seal_sha256", kind="final-seal-v3")


FINALIZATION_RECEIPT_KEYS = frozenset({
    "schema_version", "contract_sha256", "seal_ref", "completion_refs", "final_result_ref",
    "finalization_id", "finalization_sha256",
})


def finalize_finalization_receipt(payload_without_derived: Mapping[str, Any]) -> dict[str, Any]:
    return _finalize(payload_without_derived, id_field="finalization_id", hash_field="finalization_sha256",
                     kind="finalization-receipt-v3", validator=validate_finalization_receipt)


def validate_finalization_receipt(receipt: Mapping[str, Any], contract: Mapping[str, Any] | None = None,
                                  completions: Sequence[Mapping[str, Any]] | None = None) -> None:
    root = _exact(receipt, FINALIZATION_RECEIPT_KEYS, "finalization receipt")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("finalization receipt schema_version must be 3")
    contract_sha = _sha(root["contract_sha256"], "finalization receipt.contract_sha256")
    validate_artifact_ref(root["seal_ref"], "finalization receipt.seal_ref")
    refs = root["completion_refs"]
    if not isinstance(refs, Mapping) or not refs:
        raise ContractError("finalization receipt.completion_refs must be a non-empty arm map")
    for arm, ref in refs.items():
        _identifier(arm, f"finalization receipt.completion_refs key {arm!r}")
        validate_artifact_ref(ref, f"finalization receipt.completion_refs.{arm}")
    validate_artifact_ref(root["final_result_ref"], "finalization receipt.final_result_ref")
    if contract is not None:
        validate_execution_contract(contract)
        if contract["contract_sha256"] != contract_sha or set(refs) != set(contract["arms"]):
            raise ContractError("finalization receipt differs from execution contract arm set")
    if completions is not None:
        if not isinstance(completions, Sequence) or isinstance(completions, (str, bytes)):
            raise ContractError("completions must be a sequence")
        by_arm: dict[str, Mapping[str, Any]] = {}
        for completion in completions:
            validate_completion_receipt(completion, contract)
            arm = completion["arm"]
            if arm in by_arm:
                raise ContractError("finalization contains duplicate arm completion")
            by_arm[arm] = completion
        if set(by_arm) != set(refs):
            raise ContractError("finalization completion set is incomplete")
        for arm, completion in by_arm.items():
            if completion["contract_sha256"] != contract_sha:
                raise ContractError("finalization completion identity differs")
            validate_artifact_binding(refs[arm], completion, f"finalization completion_refs.{arm}")
    _validate_content(root, id_field="finalization_id", hash_field="finalization_sha256", kind="finalization-receipt-v3")


def finalization_decision(existing: Mapping[str, Any] | None, candidate: Mapping[str, Any]) -> str:
    """Create-only CAS decision for the terminal receipt."""
    validate_finalization_receipt(candidate)
    if existing is None:
        return "publish"
    validate_finalization_receipt(existing)
    if existing["contract_sha256"] != candidate["contract_sha256"]:
        raise ContractError("existing finalization belongs to another execution contract")
    if canonical_json(existing) == canonical_json(candidate):
        return "already_finalized"
    raise ContractError("conflicting finalization receipt already exists")


