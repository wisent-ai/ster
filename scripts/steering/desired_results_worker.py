#!/usr/bin/env python3
"""Generation-pinned Stado adapter for one CalibrationManifestV3 attempt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
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
runner = _load_sibling("desired_results_runner")
WorkerError = contract.ContractError


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerError(f"{label} must be a JSON object")
    if contract.canonical_json(value) != data:
        raise WorkerError(f"{label} is not in canonical JSON form")
    return value


def _load_manifest(store: Any, uri: str, generation: str) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(uri, str) or not uri:
        raise WorkerError("calibration manifest URI must be non-empty")
    if not isinstance(generation, str) or not generation:
        raise WorkerError("calibration manifest generation must be non-empty")
    data, observed = store.read(uri, generation)
    if str(observed) != generation:
        raise WorkerError("calibration manifest generation drift")
    manifest = _json_object(data, "calibration manifest")
    contract.validate_calibration_manifest(manifest)
    ref = contract.artifact_ref(
        uri, generation, str(len(data)), hashlib.sha256(data).hexdigest()
    )
    contract.validate_artifact_binding(ref, manifest)
    return manifest, ref


def _attempt_prefix(manifest: Mapping[str, Any], attempt: int) -> str:
    namespace = manifest["output_namespace"].rstrip("/")
    digest = manifest["manifest_sha256"]
    return f"{namespace}/calibration-v3/{digest}/attempt-{attempt}"


def _receipt_uri(manifest: Mapping[str, Any], attempt: int, state: str) -> str:
    if state not in contract.CALIBRATION_STATES:
        raise WorkerError(f"invalid calibration receipt state {state!r}")
    return f"{_attempt_prefix(manifest, attempt)}/{state}.json"


def _create_once_with_status(store: Any, uri: str, payload: Mapping[str, Any]) -> tuple[dict[str, str], bool]:
    """Create immutable bytes and report whether this process won creation."""
    data = contract.canonical_json(payload)
    expected_sha = hashlib.sha256(data).hexdigest()
    created = True
    try:
        ref = store.create(uri, data, content_type="application/json")
    except Exception:
        # Creation is the ownership boundary. An identical object observed
        # after a failed create belongs to the winner, never this process.
        if not store.exists(uri):
            raise
        existing, generation = store.read(uri)
        if existing != data:
            raise WorkerError(f"conflicting immutable object won create race: {uri}")
        ref = {"uri": uri, "generation": str(generation),
               "size": str(len(existing)), "sha256": expected_sha}
        created = False
    normalized = contract.validate_artifact_ref(ref, "created object")
    expected = {"uri": uri, "generation": normalized["generation"],
                "size": str(len(data)), "sha256": expected_sha}
    if normalized != expected:
        raise WorkerError("store returned an incorrect ArtifactRef")
    observed, generation = store.read(uri, normalized["generation"])
    if observed != data or str(generation) != normalized["generation"]:
        raise WorkerError("created object failed immutable read-back")
    return normalized, created


def _create_once(store: Any, uri: str, payload: Mapping[str, Any]) -> dict[str, str]:
    ref, _ = _create_once_with_status(store, uri, payload)
    return ref


def _base_receipt(manifest: Mapping[str, Any], manifest_ref: Mapping[str, Any],
                  attempt: int, state: str, runtime_evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "manifest_ref": dict(manifest_ref),
        "manifest_sha256": manifest["manifest_sha256"],
        "attempt": attempt,
        "attempt_id": contract.calibration_attempt_id(manifest["manifest_sha256"], attempt),
        "state": state,
        "runtime_evidence": dict(runtime_evidence),
        "runtime_evidence_sha256": contract.runtime_evidence_sha256(runtime_evidence),
    }


def _publish_phase(store: Any, manifest: Mapping[str, Any], manifest_ref: Mapping[str, Any],
                   attempt: int, state: str, runtime_evidence: Mapping[str, Any],
                   evidence: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    receipt = contract.finalize_calibration_phase_receipt({
        **_base_receipt(manifest, manifest_ref, attempt, state, runtime_evidence),
        "evidence": dict(evidence),
    })
    contract.validate_calibration_receipt(receipt)
    if previous is not None:
        contract.validate_calibration_transition(previous, receipt)
    _create_once(store, _receipt_uri(manifest, attempt, state), receipt)
    return receipt

def _acquire_claim(store: Any, manifest: Mapping[str, Any], manifest_ref: Mapping[str, Any],
                   attempt: int, runtime_evidence: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    claim = contract.finalize_calibration_phase_receipt({
        **_base_receipt(manifest, manifest_ref, attempt, "claim", runtime_evidence),
        "evidence": {"attempt_id": contract.calibration_attempt_id(manifest["manifest_sha256"], attempt)},
    })
    contract.validate_calibration_receipt(claim)
    _, owner = _create_once_with_status(store, _receipt_uri(manifest, attempt, "claim"), claim)
    return claim, owner


def _recover_completed_receipt(store: Any, manifest: Mapping[str, Any], attempt: int) -> dict[str, Any] | None:
    uri = _receipt_uri(manifest, attempt, "success")
    if not store.exists(uri):
        return None
    data, _ = store.read(uri)
    receipt = _json_object(data, "existing calibration success receipt")
    contract.validate_calibration_success_receipt(receipt, manifest)
    return receipt


def _safe_error(exc: BaseException, retryable: bool) -> dict[str, Any]:
    kind = type(exc).__name__
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "_", kind)[:80] or "Error"
    digest = hashlib.sha256(str(exc).encode("utf-8", "replace")).hexdigest()
    return {"type": normalized, "message": f"redacted-sha256:{digest}", "retryable": retryable}


def _publish_failure(store: Any, manifest: Mapping[str, Any], manifest_ref: Mapping[str, Any],
                     attempt: int, runtime_evidence: Mapping[str, Any], exc: BaseException,
                     previous: Mapping[str, Any] | None, *, publication_failed: bool = False) -> dict[str, Any]:
    retryable = attempt < contract.MAX_CALIBRATION_ATTEMPTS and (
        publication_failed or not isinstance(exc, (contract.ContractError, ValueError, TypeError))
    )
    receipt = contract.finalize_calibration_failure_receipt({
        **_base_receipt(manifest, manifest_ref, attempt, "failure", runtime_evidence),
        "error": _safe_error(exc, retryable),
    })
    contract.validate_calibration_failure_receipt(receipt)
    if previous is not None:
        contract.validate_calibration_transition(previous, receipt)
    _create_once(store, _receipt_uri(manifest, attempt, "failure"), receipt)
    return receipt


def _validate_runner_result(result: Any, manifest: Mapping[str, Any],
                            runtime_evidence: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(result, Mapping) or set(result) != {
        "selected_config", "result_ref", "runtime_evidence", "result"
    }:
        raise WorkerError("runner returned a malformed calibration result adapter")
    if result["runtime_evidence"] != runtime_evidence:
        raise WorkerError("runner runtime evidence differs from worker evidence")
    try:
        selected = contract.validate_selected_config(
            result["selected_config"], "runner selected_config",
        )
    except contract.ContractError as exc:
        raise WorkerError(f"runner returned an invalid selected_config: {exc}") from exc
    if selected["method"] != manifest["method"]:
        raise WorkerError("runner selected_config method differs from calibration manifest")
    result_ref = contract.validate_artifact_ref(result["result_ref"], "runner result_ref")
    payload = result["result"]
    if not isinstance(payload, Mapping):
        raise WorkerError("runner result payload must be an object")
    contract.validate_artifact_binding(result_ref, payload)
    if payload.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise WorkerError("runner result binds a different calibration manifest")
    expected = manifest["target"]
    if payload.get("target") != expected or payload.get("method") != manifest["method"]:
        raise WorkerError("runner result target/method differs from manifest")
    if payload.get("revisions") != manifest["revisions"]:
        raise WorkerError("runner result revisions differ from manifest")
    if payload.get("fit_support") != manifest["support"]["train"] or payload.get("selection_support") != manifest["support"]["validation"]:
        raise WorkerError("runner result support differs from manifest")
    if payload.get("test_reads") != 0 or payload.get("test_pair_ids_read") != []:
        raise WorkerError("calibration runner reported held-out test access")
    if payload.get("runtime_evidence") != runtime_evidence:
        raise WorkerError("runner result runtime evidence differs")
    if payload.get("selected_config") != selected:
        raise WorkerError("runner selected_config differs from published result")
    return dict(selected), result_ref


def run_attempt(store: Any, manifest_uri: str, manifest_generation: str,
                attempt_number: int) -> dict[str, Any]:
    manifest, manifest_ref = _load_manifest(store, manifest_uri, manifest_generation)
    contract.calibration_attempt_id(manifest["manifest_sha256"], attempt_number)
    runtime_evidence = contract.observe_runtime_evidence()
    expected_revision = manifest["runtime"]["revision"]
    if (runtime_evidence["runtime_revision"] != expected_revision or
            runtime_evidence["runtime_revision"] != manifest["revisions"]["code"] or
            runtime_evidence["device"] != manifest["runtime"]["device"] or
            runtime_evidence["device"] != manifest["calibration_policy"]["options"]["device"]):
        raise WorkerError("observed detached revision/device differs from calibration manifest policy")
    claim = prepared = running = None
    publication_in_progress = False
    try:
        claim, owner = _acquire_claim(
            store, manifest, manifest_ref, attempt_number, runtime_evidence,
        )
        if not owner:
            completed = _recover_completed_receipt(store, manifest, attempt_number)
            return completed if completed is not None else claim
        prepared = _publish_phase(
            store, manifest, manifest_ref, attempt_number, "prepared", runtime_evidence,
            {"manifest_verified": True, "test_support_exposed": False}, claim,
        )
        running = _publish_phase(
            store, manifest, manifest_ref, attempt_number, "running", runtime_evidence,
            {"method": manifest["method"], "test_reads": 0}, prepared,
        )
        publication_in_progress = False
        result = runner._run_calibration(
            store, manifest_uri, manifest_generation, attempt_number,
            runtime_evidence=runtime_evidence,
        )
        selected_config, result_ref = _validate_runner_result(result, manifest, runtime_evidence)
        success = contract.finalize_calibration_success_receipt({
            **_base_receipt(manifest, manifest_ref, attempt_number, "success", runtime_evidence),
            "selected_config": selected_config,
            "result_ref": result_ref,
        })
        contract.validate_calibration_success_receipt(success, manifest)
        contract.validate_calibration_transition(running, success)
        publication_in_progress = True
        _create_once(store, _receipt_uri(manifest, attempt_number, "success"), success)
        return success
    except BaseException as exc:
        previous = running or prepared or claim
        return _publish_failure(
            store, manifest, manifest_ref, attempt_number, runtime_evidence, exc, previous,
            publication_failed=publication_in_progress,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--calibration-manifest-generation", required=True)
    parser.add_argument("--attempt-number", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = runner.GCSStore()
        receipt = run_attempt(
            store, args.calibration_manifest,
            args.calibration_manifest_generation, args.attempt_number,
        )
        print(contract.canonical_json(receipt).decode("ascii"))
        return 0 if receipt["state"] == "success" else 2
    except (WorkerError, OSError, RuntimeError) as exc:
        print(f"calibration worker refused execution: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
