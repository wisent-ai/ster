#!/usr/bin/env python3
"""Execute one sealed schema-v3 final-test target wave with an injected adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.steering import desired_results_execution_contract as execution
from scripts.steering.desired_results_final_test import (
    FinalTestError, GCSStore, create_json_once, publish_json, read_completion_lineage,
    read_ref, validate_staged_result,
)

SCHEMA_VERSION = execution.SCHEMA_VERSION


class WorkerError(FinalTestError):
    """A sealed target wave cannot be executed safely."""


class _ClaimNotAcquired(WorkerError):
    """The deterministic claim bytes already exist, so this caller is not owner."""


def _binding(ref: Any, document: Any, label: str) -> dict[str, str]:
    try:
        return execution.validate_artifact_binding(ref, document, label)
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc


def _read_bound(store: Any, ref: Mapping[str, Any], label: str) -> tuple[Any, dict[str, str]]:
    document, normalized = read_ref(store, ref, label)
    return document, _binding(normalized, document, label)


def _load_sealed_wave(store: Any, seal_ref: Mapping[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, str]
]:
    """Read and bind the entire control graph before any held-out data read."""
    seal, normalized_seal_ref = _read_bound(store, seal_ref, "final seal")
    try:
        execution.validate_final_seal(seal)
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    contract, normalized_contract_ref = _read_bound(store, seal["contract_ref"], "execution contract")
    if contract != seal["contract"]:
        raise WorkerError("final seal contract differs from referenced contract bytes")
    try:
        if normalized_contract_ref != execution.validate_artifact_ref(seal["contract_ref"]):
            raise WorkerError("execution contract ref changed")
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    seal_uri = normalized_seal_ref["uri"]
    seal_suffix = f"/seals/{seal['seal_sha256']}.json"
    if not seal_uri.startswith("gs://") or not seal_uri.endswith(seal_suffix):
        raise WorkerError("final seal ref is foreign to its canonical target namespace")
    target_prefix = seal_uri[:-len(seal_suffix)]
    if normalized_contract_ref["uri"] != f"{target_prefix}/contracts/{contract['contract_sha256']}.json":
        raise WorkerError("execution contract ref is foreign to the final seal target namespace")
    manifests: dict[str, dict[str, Any]] = {}
    for arm in contract["arms"]:
        manifest, manifest_ref = _read_bound(
            store, seal["arm_manifest_refs"][arm], f"arm manifest {arm}"
        )
        try:
            expected_manifest_ref = execution.validate_artifact_ref(seal["arm_manifest_refs"][arm])
        except execution.ContractError as exc:
            raise WorkerError(str(exc)) from exc
        if manifest_ref != expected_manifest_ref:
            raise WorkerError(f"arm manifest ref changed for {arm}")
        expected_manifest_uri = f"{target_prefix}/arms/{arm}/{manifest['manifest_sha256']}.json"
        if manifest_ref["uri"] != expected_manifest_uri:
            raise WorkerError(f"arm manifest ref is foreign to the final seal target namespace for {arm}")
        manifests[arm] = manifest
    try:
        execution.validate_final_seal(seal, manifests)
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    return seal, contract, manifests, normalized_seal_ref


def _model_identity(adapter: Any, model: Any) -> dict[str, str]:
    if not hasattr(adapter, "model_identity"):
        identity = {
            "model_revision": getattr(model, "resolved_model_revision", None),
            "tokenizer_revision": getattr(model, "resolved_tokenizer_revision", None),
        }
    else:
        identity = adapter.model_identity(model)
    if not isinstance(identity, Mapping) or set(identity) != {"model_revision", "tokenizer_revision"}:
        raise WorkerError("adapter model_identity must return exact model/tokenizer revision pair")
    if not all(isinstance(identity[key], str) and identity[key] for key in identity):
        raise WorkerError("adapter returned unresolved model/tokenizer revision")
    return dict(identity)

ADAPTER_RUNTIME_EVIDENCE_KEYS = frozenset({
    "model_revision", "tokenizer_revision", "activation_revision",
    "runtime_revision", "device",
})


def _validated_adapter_evaluation(value: Any, expected: Mapping[str, str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"scores", "runtime_evidence"}:
        raise WorkerError("adapter evaluation must return exact scores/runtime_evidence fields")
    evidence = value["runtime_evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != ADAPTER_RUNTIME_EVIDENCE_KEYS:
        raise WorkerError("adapter evaluation runtime_evidence fields are not exact")
    if any(not isinstance(evidence[key], str) or not evidence[key] for key in evidence):
        raise WorkerError("adapter evaluation runtime_evidence values must be resolved strings")
    if dict(evidence) != dict(expected):
        raise WorkerError("adapter returned stale or foreign runtime/activation evidence")
    scores = value["scores"]
    if not isinstance(scores, Mapping):
        raise WorkerError("adapter evaluation scores must be an object")
    return scores


def _selected_config(contract: Mapping[str, Any], arm: str) -> Mapping[str, Any] | None:
    if arm == "baseline":
        return None
    matches = [receipt for receipt in contract["calibration_receipts"]
               if receipt["selected_config"].get("method") == arm]
    if len(matches) != 1:
        raise WorkerError(f"arm {arm} lacks exactly one selected calibration")
    return matches[0]["selected_config"]


def _read_selected_artifact(store: Any, manifest: Mapping[str, Any], selected: Any) -> Any:
    if selected is None:
        return None
    document, _ = _read_bound(store, manifest["selected_config_ref"], "selected config")
    if document == selected:
        return document
    if not isinstance(document, Mapping) or document.get("selected_config") != selected:
        raise WorkerError("selected config artifact differs from sealed selected_config")
    return document


def _read_route_artifacts(store: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for index, route in enumerate(manifest["support_refs"]):
        completion, completion_ref = _read_bound(
            store, route["completion_ref"], f"support route {index} completion"
        )
        proof, proof_ref = _read_bound(store, route["proof_ref"], f"support route {index} proof")
        routes.append({"strategy": route["strategy"], "layer": route["layer"],
                       "completion": completion, "completion_ref": completion_ref,
                       "proof": proof, "proof_ref": proof_ref})
    return routes


def _pair_rows(document: Any) -> list[Mapping[str, Any]]:
    rows = document.get("pairs") if isinstance(document, Mapping) else None
    if not isinstance(rows, list):
        raise WorkerError("pair_texts artifact must contain a pairs list")
    return rows


def _test_rows(contract: Mapping[str, Any], pair_document: Any) -> list[Mapping[str, Any]]:
    expected = contract["target_manifest"]["support"]["splits"]["test"]
    by_pair: dict[int, Mapping[str, Any]] = {}
    for row in _pair_rows(pair_document):
        if not isinstance(row, Mapping) or type(row.get("pair_id")) is not int:
            raise WorkerError("pair_texts contains a malformed pair row")
        if row["pair_id"] in by_pair:
            raise WorkerError("pair_texts contains duplicate pair_id")
        by_pair[row["pair_id"]] = row
    selected: list[Mapping[str, Any]] = []
    for identity in expected:
        row = by_pair.get(identity["pair_id"])
        if row is None or row.get("stable_id") != identity["stable_id"]:
            raise WorkerError("pair_texts does not preserve exact held-out support identity")
        selected.append(row)
    if len(selected) != len(expected):
        raise WorkerError("pair_texts held-out support count differs")
    return selected


def _attempt_payload(
    contract: Mapping[str, Any], manifest: Mapping[str, Any], attempt: int,
    phase: str, evidence: Mapping[str, Any], *, staged: Any = None,
    publication: Any = None, quarantined: bool = False,
) -> dict[str, Any]:
    arm_sha = manifest["manifest_sha256"]
    post_token = execution.ATTEMPT_PHASES.index(phase) >= execution.ATTEMPT_PHASES.index("test_token_consumed")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "arm_manifest_sha256": arm_sha,
        "arm": manifest["arm"],
        "attempt": attempt,
        "attempt_id": execution.attempt_id(arm_sha, attempt),
        "phase": phase,
        "test_token_id": execution.test_token_id(arm_sha) if post_token else None,
        "evidence": dict(evidence),
        "staged_result_ref": staged,
        "publication_ref": publication,
        "quarantined": quarantined,
    }


def _publish_attempt(
    store: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any], attempt: int,
    phase: str, evidence: Mapping[str, Any], *, previous: Mapping[str, Any] | None = None,
    staged: Any = None, publication: Any = None, quarantined: bool = False,
    exclusive: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        receipt = execution.finalize_attempt_receipt(_attempt_payload(
            contract, manifest, attempt, phase, evidence, staged=staged,
            publication=publication, quarantined=quarantined,
        ))
        execution.validate_attempt_receipt(receipt, contract, manifest)
        if previous is not None:
            execution.validate_attempt_transition(previous, receipt)
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    uri = manifest["output_prefix"] + f"attempts/{attempt}/{phase}.json"
    if exclusive:
        ref, acquired = create_json_once(store, uri, receipt)
        if not acquired:
            raise _ClaimNotAcquired(
                f"claim already exists for {manifest['arm']} attempt {attempt}; caller is not owner"
            )
    else:
        ref = publish_json(store, uri, receipt)
    _binding(ref, receipt, f"attempt receipt {manifest['arm']} {phase}")
    return receipt, ref


def _normalize_scores(
    contract: Mapping[str, Any], manifest: Mapping[str, Any], scores: Any,
    test_rows: Sequence[Mapping[str, Any]], token_id: str,
) -> dict[str, Any]:
    if not isinstance(scores, Mapping):
        raise WorkerError("adapter score_batch must return an object")
    required = contract["target_manifest"]["evaluation"]["required_outputs"]
    missing = [name for name in required if name not in scores]
    if missing:
        raise WorkerError(f"adapter result misses required outputs: {missing}")
    predictions = scores.get("predictions")
    if predictions is not None:
        if not isinstance(predictions, list) or len(predictions) != len(test_rows):
            raise WorkerError("predictions do not preserve exact held-out batch size")
        for expected, prediction in zip(test_rows, predictions):
            if not isinstance(prediction, Mapping) or prediction.get("pair_id") != expected["pair_id"] or prediction.get("stable_id") != expected["stable_id"]:
                raise WorkerError("predictions do not preserve ordered pair identity")
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "arm_manifest_sha256": manifest["manifest_sha256"],
        "arm": manifest["arm"],
        "target": dict(contract["target"]),
        "revisions": dict(contract["revisions"]),
        "runtime_evidence": contract["runtime_evidence"],
        "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
        "pair_texts_ref": manifest["pair_texts_ref"],
        "evaluator_ref": manifest["evaluator_ref"],
        "support_refs": manifest["support_refs"],
        "test_token_id": token_id,
        "test_token_consumptions": 1,
        "test_pair_count": len(test_rows),
        "scores": dict(scores),
    }
    try:
        return validate_staged_result(result, contract, manifest)
    except FinalTestError as exc:
        raise WorkerError(str(exc)) from exc


def _quarantine(
    store: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any], attempt: int,
    evidence: Mapping[str, Any], previous: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    return _publish_attempt(store, contract, manifest, attempt, "completed", evidence,
                            previous=previous, quarantined=True)


def _execute_arm(
    store: Any, adapter: Any, model: Any, contract: Mapping[str, Any],
    manifest: Mapping[str, Any], evaluator: Mapping[str, Any], attempt: int,
    common_evidence: Mapping[str, Any], adapter_runtime_evidence: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    arm = manifest["arm"]
    selected = _selected_config(contract, arm)
    evidence = dict(common_evidence)
    evidence.update({"arm": arm, "pair_texts_ref": manifest["pair_texts_ref"],
                     "evaluator_ref": manifest["evaluator_ref"],
                     "support_refs": manifest["support_refs"]})
    try:
        current, _ = _publish_attempt(
            store, contract, manifest, attempt, "claimed", evidence, exclusive=True,
        )
    except _ClaimNotAcquired as exc:
        existing = _existing_completion(store, contract, manifest)
        if existing is not None:
            return existing
        raise WorkerError(
            f"claim already exists for {arm} attempt {attempt}; held-out read forbidden"
        ) from exc
    selected_artifact = _read_selected_artifact(store, manifest, selected)
    route_artifacts = _read_route_artifacts(store, manifest)
    prepare_routes = getattr(adapter, "prepare_routes", None)
    if not callable(prepare_routes):
        raise WorkerError("adapter must implement activation-aware prepare_routes")
    route_artifacts = prepare_routes(
        route_artifacts, activation_revision=contract["revisions"]["activation"],
    )
    current, _ = _publish_attempt(store, contract, manifest, attempt, "inputs_verified", evidence, previous=current)
    current, _ = _publish_attempt(store, contract, manifest, attempt, "fit_complete", evidence, previous=current)
    current, _ = _publish_attempt(store, contract, manifest, attempt, "test_token_consumed", evidence, previous=current)
    token_id = current["test_token_id"]
    installed = False
    try:
        try:
            # This is intentionally the first held-out pair-text read for the arm.
            pair_document, pair_ref = _read_bound(store, manifest["pair_texts_ref"], f"pair texts {arm}")
            if pair_ref != execution.validate_artifact_ref(manifest["pair_texts_ref"]):
                raise WorkerError("pair_texts ref changed")
            test_rows = _test_rows(contract, pair_document)
            support = adapter.load_support(
                arm=arm, pair_texts=pair_document, route_payloads=route_artifacts,
                test_rows=test_rows,
            )
            adapter.install(model=model, arm=arm, selected_config=selected,
                            selected_artifact=selected_artifact, support=support)
            installed = True
            evaluation = adapter.score_batch(
                model=model, arm=arm, support=support, test_rows=test_rows,
                evaluator=evaluator, activation_revision=contract["revisions"]["activation"],
            )
            scores = _validated_adapter_evaluation(evaluation, adapter_runtime_evidence)
            result = _normalize_scores(contract, manifest, scores, test_rows, token_id)
        finally:
            if installed:
                adapter.remove(model=model, arm=arm)
    except BaseException as exc:
        _quarantine(store, contract, manifest, attempt, evidence, current)
        if isinstance(exc, WorkerError):
            raise
        raise WorkerError(f"post-token evaluation failed for {arm}: {exc}") from exc
    staged_ref = publish_json(store, manifest["output_prefix"] + f"attempts/{attempt}/staged-result.json", result)
    current, _ = _publish_attempt(store, contract, manifest, attempt, "evaluated", evidence,
                                  previous=current, staged=staged_ref)
    publication_ref = publish_json(store, manifest["output_prefix"] + "result.json", result)
    current, _ = _publish_attempt(store, contract, manifest, attempt, "artifacts_published", evidence,
                                  previous=current, staged=staged_ref, publication=publication_ref)
    current, current_ref = _publish_attempt(store, contract, manifest, attempt, "completed", evidence,
                                            previous=current, staged=staged_ref, publication=publication_ref)
    return _publish_completion(store, contract, manifest, current, current_ref)


def _read_optional_uri(store: Any, uri: str, label: str) -> tuple[Any, dict[str, str]] | None:
    try:
        raw = store.read(uri, None)
    except Exception as exc:
        missing = str(exc).lower().startswith("object missing:")
        if (isinstance(exc, (KeyError, FileNotFoundError)) or type(exc).__name__ == "NotFound"
                or getattr(exc, "code", None) == 404 or missing):
            return None
        raise WorkerError(f"cannot inspect {label}: {exc}") from exc
    if not isinstance(raw, tuple) or len(raw) != 2 or not isinstance(raw[0], bytes):
        raise WorkerError(f"store returned malformed bytes for {label}")
    generation = raw[1].get("generation") if isinstance(raw[1], Mapping) else raw[1]
    import hashlib
    ref = execution.artifact_ref(uri, str(generation), str(len(raw[0])), hashlib.sha256(raw[0]).hexdigest())
    document, normalized = read_ref(store, ref, label)
    return document, _binding(normalized, document, label)


def _discover_attempts(
    store: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, str]]]:
    found: list[tuple[dict[str, Any], dict[str, str]]] = []
    for attempt_number in range(1, execution.MAX_PRE_TEST_ATTEMPTS + 1):
        previous: Mapping[str, Any] | None = None
        for phase in execution.ATTEMPT_PHASES:
            uri = manifest["output_prefix"] + f"attempts/{attempt_number}/{phase}.json"
            item = _read_optional_uri(store, uri, f"existing {manifest['arm']} {phase}")
            if item is None:
                continue
            receipt, ref = item
            try:
                execution.validate_attempt_receipt(receipt, contract, manifest)
                if receipt["attempt"] != attempt_number or receipt["phase"] != phase:
                    raise execution.ContractError("attempt receipt path differs from its identity")
                if previous is not None:
                    execution.validate_attempt_transition(previous, receipt)
            except execution.ContractError as exc:
                raise WorkerError(f"existing attempt receipt drift: {exc}") from exc
            previous = receipt
            found.append((receipt, ref))
    return found


def _publish_completion(
    store: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any],
    completed: Mapping[str, Any], completed_ref: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    durable_completed, normalized_completed_ref = _read_bound(
        store, completed_ref, f"completed attempt {manifest['arm']}",
    )
    if durable_completed != completed:
        raise WorkerError("completed attempt differs from its referenced bytes")
    try:
        expected_completed_ref = execution.validate_artifact_ref(completed_ref)
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    if normalized_completed_ref != expected_completed_ref:
        raise WorkerError("completed attempt ref changed")
    expected_attempt_uri = (
        manifest["output_prefix"] + f"attempts/{completed['attempt']}/completed.json"
    )
    expected_stage_uri = (
        manifest["output_prefix"] + f"attempts/{completed['attempt']}/staged-result.json"
    )
    expected_publication_uri = manifest["output_prefix"] + "result.json"
    if normalized_completed_ref["uri"] != expected_attempt_uri:
        raise WorkerError("completed attempt ref is foreign to its arm/attempt lineage")
    if completed["staged_result_ref"]["uri"] != expected_stage_uri:
        raise WorkerError("staged result ref is foreign to its arm/attempt lineage")
    if completed["publication_ref"]["uri"] != expected_publication_uri:
        raise WorkerError("publication ref is foreign to its sealed arm")
    staged_result, normalized_staged_ref = _read_bound(
        store, completed["staged_result_ref"], f"staged result {manifest['arm']}",
    )
    published_result, normalized_publication_ref = _read_bound(
        store, completed["publication_ref"], f"published result {manifest['arm']}",
    )
    try:
        expected_staged_ref = execution.validate_artifact_ref(completed["staged_result_ref"])
        expected_publication_ref = execution.validate_artifact_ref(completed["publication_ref"])
        staged_result = validate_staged_result(staged_result, contract, manifest)
        published_result = validate_staged_result(published_result, contract, manifest, "published result")
    except (execution.ContractError, FinalTestError) as exc:
        raise WorkerError(str(exc)) from exc
    if normalized_staged_ref != expected_staged_ref or normalized_publication_ref != expected_publication_ref:
        raise WorkerError("completed output ref changed")
    if staged_result != published_result:
        raise WorkerError("published result differs from durable staged result")
    try:
        completion = execution.finalize_completion_receipt({
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": contract["contract_sha256"],
            "arm_manifest_sha256": manifest["manifest_sha256"],
            "arm": manifest["arm"],
            "attempt_id": completed["attempt_id"],
            "attempt_receipt_ref": normalized_completed_ref,
            "staged_result_ref": normalized_staged_ref,
            "publication_ref": normalized_publication_ref,
        })
        execution.validate_completion_receipt(completion, contract, manifest, completed)
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    ref = publish_json(store, manifest["output_prefix"] + "completion.json", completion)
    try:
        verified, verified_ref, _, _ = read_completion_lineage(
            store, ref, contract, manifest, f"completion {manifest['arm']}",
        )
    except FinalTestError as exc:
        raise WorkerError(str(exc)) from exc
    return verified, verified_ref


def _existing_completion(
    store: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    item = _read_optional_uri(store, manifest["output_prefix"] + "completion.json",
                              f"existing completion {manifest['arm']}")
    if item is None:
        return None
    _, ref = item
    try:
        completion, normalized_ref, _, _ = read_completion_lineage(
            store, ref, contract, manifest, f"existing completion {manifest['arm']}",
        )
    except FinalTestError as exc:
        raise WorkerError(str(exc)) from exc
    return completion, normalized_ref


def _resume_publication(
    store: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any],
    latest: Mapping[str, Any], latest_ref: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    current, current_ref = dict(latest), dict(latest_ref)
    if current["phase"] == "completed":
        if current["quarantined"]:
            raise WorkerError(f"arm {manifest['arm']} is terminally quarantined")
        return _publish_completion(store, contract, manifest, current, current_ref)
    staged_ref = current["staged_result_ref"]
    if staged_ref is None:
        quarantined, _ = _quarantine(store, contract, manifest, current["attempt"], current["evidence"], current)
        raise WorkerError(f"arm {manifest['arm']} consumed its test token without durable staged output: {quarantined['receipt_id']}")
    expected_stage_uri = (
        manifest["output_prefix"] + f"attempts/{current['attempt']}/staged-result.json"
    )
    if staged_ref["uri"] != expected_stage_uri:
        raise WorkerError("staged result ref is foreign to its arm/attempt lineage")
    staged_result, normalized_stage = _read_bound(store, staged_ref, f"staged result {manifest['arm']}")
    try:
        expected_stage = execution.validate_artifact_ref(staged_ref)
        staged_result = validate_staged_result(staged_result, contract, manifest)
    except (execution.ContractError, FinalTestError) as exc:
        raise WorkerError(str(exc)) from exc
    if normalized_stage != expected_stage:
        raise WorkerError("staged result ref changed")
    publication_ref = current["publication_ref"]
    if publication_ref is not None and publication_ref["uri"] != manifest["output_prefix"] + "result.json":
        raise WorkerError("publication ref is foreign to its sealed arm")
    if publication_ref is None:
        publication_ref = publish_json(store, manifest["output_prefix"] + "result.json", staged_result)
    else:
        published, normalized_publication = _read_bound(store, publication_ref, f"published result {manifest['arm']}")
        try:
            expected_publication = execution.validate_artifact_ref(publication_ref)
            published = validate_staged_result(published, contract, manifest, "published result")
        except (execution.ContractError, FinalTestError) as exc:
            raise WorkerError(str(exc)) from exc
        if published != staged_result or normalized_publication != expected_publication:
            raise WorkerError("published result differs from durable staged result")
    if current["phase"] == "evaluated":
        current, current_ref = _publish_attempt(
            store, contract, manifest, current["attempt"], "artifacts_published", current["evidence"],
            previous=current, staged=staged_ref, publication=publication_ref,
        )
    if current["phase"] == "artifacts_published":
        current, current_ref = _publish_attempt(
            store, contract, manifest, current["attempt"], "completed", current["evidence"],
            previous=current, staged=staged_ref, publication=publication_ref,
        )
    return _publish_completion(store, contract, manifest, current, current_ref)


def _select_sealed_arm(
    seal: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]],
    manifest_ref: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve an arm only through the exact manifest ref carried by the loaded seal."""
    try:
        normalized_manifest_ref = execution.validate_artifact_ref(
            manifest_ref, "arm manifest ref",
        )
    except execution.ContractError as exc:
        raise WorkerError(str(exc)) from exc
    matching_arms = [
        arm for arm, sealed_ref in seal["arm_manifest_refs"].items()
        if sealed_ref == normalized_manifest_ref
    ]
    if len(matching_arms) != 1:
        raise WorkerError("arm manifest ref is not reachable from the exact final seal")
    arm = matching_arms[0]
    manifest = manifests.get(arm)
    if not isinstance(manifest, Mapping):
        raise WorkerError("sealed arm manifest is missing from the loaded control graph")
    return dict(manifest), normalized_manifest_ref


def _observed_runtime(
    contract: Mapping[str, Any], device: str,
    runtime_observer: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        observed = execution.validate_runtime_evidence(runtime_observer())
    except Exception as exc:
        raise WorkerError(f"cannot observe exact runtime evidence: {exc}") from exc
    if (observed != contract["runtime_evidence"] or
            execution.runtime_evidence_sha256(observed) != contract["runtime_evidence_sha256"] or
            observed["runtime_revision"] != contract["revisions"]["code"] or
            observed["runtime_revision"] != contract["revisions"]["runtime"] or
            observed["device"] != contract["calibration_policy"]["options"]["device"] or
            device != observed["device"]):
        raise WorkerError("observed detached revision/device/package evidence differs from sealed policy")
    return observed


def _execute_wave(
    store: Any, seal_ref: Mapping[str, Any], adapter: Any, device: str, *, attempt: int,
    runtime_observer: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    seal, contract, manifests, normalized_seal_ref = _load_sealed_wave(store, seal_ref)
    if type(attempt) is not int or not 1 <= attempt <= execution.MAX_PRE_TEST_ATTEMPTS:
        raise WorkerError("attempt is outside the sealed retry policy")
    observed = _observed_runtime(contract, device, runtime_observer)
    runtime_device = observed["device"]

    completions: dict[str, dict[str, Any]] = {}
    completion_refs: dict[str, dict[str, str]] = {}
    fresh: list[tuple[str, int]] = []
    for arm in contract["arms"]:
        manifest = manifests[arm]
        existing = _existing_completion(store, contract, manifest)
        if existing is not None:
            completions[arm], completion_refs[arm] = existing
            continue
        history = _discover_attempts(store, contract, manifest)
        if not history:
            fresh.append((arm, attempt))
            continue
        receipts = [item[0] for item in history]
        try:
            decision = execution.retry_decision(receipts)
        except execution.ContractError as exc:
            raise WorkerError(str(exc)) from exc
        latest, latest_ref = max(
            history,
            key=lambda item: (item[0]["attempt"], execution.ATTEMPT_PHASES.index(item[0]["phase"])),
        )
        if decision in {"complete", "resume_publication", "quarantine"}:
            completion, ref = _resume_publication(store, contract, manifest, latest, latest_ref)
            completions[arm], completion_refs[arm] = completion, ref
            continue
        next_attempt = max(attempt, latest["attempt"] + 1)
        if next_attempt > execution.MAX_PRE_TEST_ATTEMPTS:
            raise WorkerError(f"arm {arm} exhausted pre-test attempts")
        fresh.append((arm, next_attempt))

    if fresh:
        evaluator, evaluator_ref = _read_bound(store, contract["evaluator_ref"], "evaluator")
        if evaluator != contract["evaluator"] or evaluator_ref != execution.validate_artifact_ref(contract["evaluator_ref"]):
            raise WorkerError("evaluator artifact differs from sealed evaluator")
        revisions = contract["revisions"]
        try:
            model = adapter.load_model(
                target=contract["target"], model_revision=revisions["model"],
                tokenizer_revision=revisions["tokenizer"], device=runtime_device,
            )
        except Exception as exc:
            raise WorkerError(f"adapter model load failed: {exc}") from exc
        identity = _model_identity(adapter, model)
        if identity != {"model_revision": revisions["model"], "tokenizer_revision": revisions["tokenizer"]}:
            try:
                adapter.close(model)
            finally:
                raise WorkerError("loaded model/tokenizer revision pair differs from sealed contract")
        adapter_evidence = {
            "model_revision": identity["model_revision"],
            "tokenizer_revision": identity["tokenizer_revision"],
            "activation_revision": revisions["activation"],
            "runtime_revision": observed["runtime_revision"],
            "device": runtime_device,
        }
        evidence = {"seal_ref": normalized_seal_ref,
                    "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
                    **adapter_evidence}
        try:
            for arm, arm_attempt in fresh:
                completion, ref = _execute_arm(
                    store, adapter, model, contract, manifests[arm], evaluator, arm_attempt,
                    evidence, adapter_evidence,
                )
                completions[arm], completion_refs[arm] = completion, ref
        finally:
            adapter.close(model)
    return {"contract_sha256": contract["contract_sha256"], "target": dict(contract["target"]),
            "seal_ref": normalized_seal_ref, "completions": completions,
            "completion_refs": completion_refs}


def execute_wave(
    store: Any, seal_ref: Mapping[str, Any], adapter: Any, device: str, *, attempt: int = 1,
    runtime_observer: Callable[[], Mapping[str, Any]] = execution.observe_runtime_evidence,
) -> dict[str, Any]:
    """Execute/resume all target arms without ever consuming a test token twice."""
    try:
        return _execute_wave(store, seal_ref, adapter, device, attempt=attempt,
                             runtime_observer=runtime_observer)
    except WorkerError:
        raise
    except FinalTestError as exc:
        raise WorkerError(str(exc)) from exc
    except Exception as exc:
        raise WorkerError(f"final-test worker failed: {exc}") from exc

def _execute_single_arm(
    store: Any, seal_ref: Mapping[str, Any], manifest_ref: Mapping[str, Any],
    adapter: Any, device: str, *, attempt: int,
    runtime_observer: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    seal, contract, manifests, normalized_seal_ref = _load_sealed_wave(store, seal_ref)
    manifest, normalized_manifest_ref = _select_sealed_arm(seal, manifests, manifest_ref)
    if type(attempt) is not int or not 1 <= attempt <= execution.MAX_PRE_TEST_ATTEMPTS:
        raise WorkerError("attempt is outside the sealed retry policy")
    observed = _observed_runtime(contract, device, runtime_observer)
    existing = _existing_completion(store, contract, manifest)
    if existing is not None:
        completion, completion_ref = existing
        return {
            "contract_sha256": contract["contract_sha256"], "target": dict(contract["target"]),
            "seal_ref": normalized_seal_ref,
            "arm": manifest["arm"], "arm_manifest_ref": normalized_manifest_ref,
            "completion": completion, "completion_ref": completion_ref,
        }
    history = _discover_attempts(store, contract, manifest)
    arm_attempt = attempt
    if history:
        receipts = [item[0] for item in history]
        try:
            decision = execution.retry_decision(receipts)
        except execution.ContractError as exc:
            raise WorkerError(str(exc)) from exc
        latest, latest_ref = max(
            history,
            key=lambda item: (item[0]["attempt"], execution.ATTEMPT_PHASES.index(item[0]["phase"])),
        )
        if decision in {"complete", "resume_publication", "quarantine"}:
            completion, completion_ref = _resume_publication(
                store, contract, manifest, latest, latest_ref,
            )
            return {
                "contract_sha256": contract["contract_sha256"], "target": dict(contract["target"]),
                "seal_ref": normalized_seal_ref,
                "arm": manifest["arm"], "arm_manifest_ref": normalized_manifest_ref,
                "completion": completion, "completion_ref": completion_ref,
            }
        arm_attempt = max(attempt, latest["attempt"] + 1)
        if arm_attempt > execution.MAX_PRE_TEST_ATTEMPTS:
            raise WorkerError(f"arm {manifest['arm']} exhausted pre-test attempts")
    evaluator, evaluator_ref = _read_bound(store, contract["evaluator_ref"], "evaluator")
    if (evaluator != contract["evaluator"] or
            evaluator_ref != execution.validate_artifact_ref(contract["evaluator_ref"])):
        raise WorkerError("evaluator artifact differs from sealed evaluator")
    revisions = contract["revisions"]
    try:
        model = adapter.load_model(
            target=contract["target"], model_revision=revisions["model"],
            tokenizer_revision=revisions["tokenizer"], device=observed["device"],
        )
    except Exception as exc:
        raise WorkerError(f"adapter model load failed: {exc}") from exc
    identity = _model_identity(adapter, model)
    if identity != {"model_revision": revisions["model"], "tokenizer_revision": revisions["tokenizer"]}:
        try:
            adapter.close(model)
        finally:
            raise WorkerError("loaded model/tokenizer revision pair differs from sealed contract")
    adapter_evidence = {
        "model_revision": identity["model_revision"],
        "tokenizer_revision": identity["tokenizer_revision"],
        "activation_revision": revisions["activation"],
        "runtime_revision": observed["runtime_revision"],
        "device": observed["device"],
    }
    evidence = {
        "seal_ref": normalized_seal_ref,
        "arm_manifest_ref": normalized_manifest_ref,
        "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
        **adapter_evidence,
    }
    try:
        completion, completion_ref = _execute_arm(
            store, adapter, model, contract, manifest, evaluator, arm_attempt,
            evidence, adapter_evidence,
        )
    finally:
        adapter.close(model)
    return {
        "contract_sha256": contract["contract_sha256"], "target": dict(contract["target"]),
        "seal_ref": normalized_seal_ref,
        "arm": manifest["arm"], "arm_manifest_ref": normalized_manifest_ref,
        "completion": completion, "completion_ref": completion_ref,
    }


def execute_arm(
    store: Any, seal_ref: Mapping[str, Any], manifest_ref: Mapping[str, Any],
    adapter: Any, device: str, *, attempt: int = 1,
    runtime_observer: Callable[[], Mapping[str, Any]] = execution.observe_runtime_evidence,
) -> dict[str, Any]:
    """Execute or recover one arm reachable from an exact immutable final seal."""
    try:
        return _execute_single_arm(
            store, seal_ref, manifest_ref, adapter, device, attempt=attempt,
            runtime_observer=runtime_observer,
        )
    except WorkerError:
        raise
    except FinalTestError as exc:
        raise WorkerError(str(exc)) from exc
    except Exception as exc:
        raise WorkerError(f"final-test arm worker failed: {exc}") from exc



class ProductionFinalTestAdapter:
    """Wisent runtime adapter for one sealed target wave.

    The adapter owns only ephemeral local files. ``execute_wave`` supplies exact
    bound route documents; the adapter materializes only their pinned HF bytes.
    """

    def __init__(self, contract: Mapping[str, Any], device: str) -> None:
        execution.validate_execution_contract(contract)
        self.contract = contract
        self.device = device
        self.root = Path(tempfile.mkdtemp(prefix="desired-results-final-v3-"))
        self._installed: dict[str, tuple[Mapping[str, Any] | None, Mapping[str, Any]]] = {}
        self._activation_paths: dict[tuple[str, str, str, str, str, int], Path] = {}
        self._prepared_activation_revision: str | None = None

    @staticmethod
    def _exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != keys:
            actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
            raise WorkerError(f"{label} keys differ from the sealed preflight schema: {actual}")
        return value

    def _materialize_activation(self, artifact: Mapping[str, Any]) -> Path:
        identity = (
            artifact["repo_id"], artifact["repo_type"], artifact["revision"],
            artifact["path"], artifact["lfs_sha256"], artifact["size"],
        )
        existing = self._activation_paths.get(identity)
        if existing is not None:
            return existing
        try:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(
                repo_id=artifact["repo_id"], repo_type=artifact["repo_type"],
                revision=artifact["revision"], filename=artifact["path"],
            ))
        except Exception as exc:
            raise WorkerError(f"cannot download exact sealed activation artifact: {exc}") from exc
        if not source.is_file():
            raise WorkerError("exact sealed activation download did not return a file")
        activation_root = self.root / "sealed-activations"
        activation_root.mkdir(exist_ok=True)
        destination = activation_root / f"{artifact['lfs_sha256']}.safetensors"
        temporary = activation_root / f".{artifact['lfs_sha256']}.{len(self._activation_paths)}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    size += len(chunk)
                    if size > artifact["size"]:
                        raise WorkerError("sealed activation exceeds its proven byte size")
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if size != artifact["size"] or digest.hexdigest() != artifact["lfs_sha256"]:
                raise WorkerError("sealed activation bytes differ from proven LFS SHA/size")
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if (destination.stat().st_size != size or
                        hashlib.sha256(destination.read_bytes()).hexdigest() != digest.hexdigest()):
                    raise WorkerError("local sealed activation cache has conflicting bytes")
            self._activation_paths[identity] = destination
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    def _prepare_route(self, route: Mapping[str, Any], index: int) -> dict[str, Any]:
        route = self._exact_keys(
            route,
            {"strategy", "layer", "completion", "completion_ref", "proof", "proof_ref"},
            f"support route {index}",
        )
        completion = self._exact_keys(route["completion"], {
            "schema_version", "complete", "target_id", "route", "proof_ref",
            "activation_lfs_sha256", "activation_header_sha256",
        }, f"support route {index} completion")
        proof = self._exact_keys(route["proof"], {
            "schema_version", "proof_kind", "target_id", "activation_artifact", "route",
            "pair_ids", "tensor_shapes", "tensor_dtypes", "safetensors_header_length",
            "safetensors_header_sha256", "tensor_payload_downloaded",
        }, f"support route {index} proof")
        identity = {"strategy": route["strategy"], "layer": route["layer"]}
        target_id = self.contract["target"]["target_id"]
        if (completion["schema_version"] != 2 or completion["complete"] is not True or
                proof["schema_version"] != 2 or proof["proof_kind"] != "pinned_hf_safetensors_header" or
                completion["target_id"] != target_id or proof["target_id"] != target_id or
                completion["route"] != identity or proof["route"] != identity):
            raise WorkerError("sealed activation completion/proof route identity differs")
        if execution.validate_artifact_ref(completion["proof_ref"]) != route["proof_ref"]:
            raise WorkerError("sealed activation completion binds a different proof ref")
        artifact = self._exact_keys(proof["activation_artifact"], {
            "repo_id", "repo_type", "revision", "path", "lfs_sha256", "size",
        }, f"support route {index} activation artifact")
        expected_path = (
            f"activations/{self.contract['target']['model_slug']}/"
            f"{self.contract['target']['benchmark']}/{route['strategy']}/"
            f"layer_{route['layer']}.safetensors"
        )
        expected_revision = self.contract["revisions"]["activation"]
        target_revision = self.contract["target_manifest"]["revisions"]["activation_revision"]
        sha = artifact["lfs_sha256"]
        if (artifact["repo_type"] not in {"dataset", "model", "space"} or
                not isinstance(artifact["repo_id"], str) or not artifact["repo_id"] or
                artifact["revision"] != expected_revision or artifact["revision"] != target_revision or
                artifact["path"] != expected_path or
                not isinstance(sha, str) or len(sha) != 64 or
                any(character not in "0123456789abcdef" for character in sha) or
                type(artifact["size"]) is not int or artifact["size"] <= 0):
            raise WorkerError("sealed activation revision/path/LFS identity differs")
        if (completion["activation_lfs_sha256"] != sha or
                completion["activation_header_sha256"] != proof["safetensors_header_sha256"]):
            raise WorkerError("sealed activation completion and proof hashes differ")
        expected_pair_ids = list(range(self.contract["target_manifest"]["target"]["expected_pairs"]))
        if proof["pair_ids"] != expected_pair_ids:
            raise WorkerError("sealed activation proof does not cover exact canonical pair support")
        activation_path = self._materialize_activation(artifact)
        try:
            with activation_path.open("rb") as handle:
                prefix = handle.read(8)
                header_length = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else None
                header = handle.read(header_length or 0)
        except (OSError, struct.error) as exc:
            raise WorkerError(f"cannot inspect sealed activation header: {exc}") from exc
        if (header_length != proof["safetensors_header_length"] or
                hashlib.sha256(header).hexdigest() != proof["safetensors_header_sha256"]):
            raise WorkerError("sealed activation header differs from its proof")
        prepared = dict(route)
        prepared["activation_path"] = activation_path
        return prepared

    def prepare_routes(
        self, route_payloads: Sequence[Mapping[str, Any]], *, activation_revision: str,
    ) -> list[dict[str, Any]]:
        """Validate and materialize the exact activation revision before test-token use."""
        if (activation_revision != self.contract["revisions"]["activation"] or
                activation_revision != self.contract["target_manifest"]["revisions"]["activation_revision"]):
            raise WorkerError("adapter preparation activation revision differs from sealed contract")
        prepared = [self._prepare_route(route, index) for index, route in enumerate(route_payloads)]
        observed = {
            route["proof"]["activation_artifact"]["revision"] for route in prepared
        }
        if observed and observed != {activation_revision}:
            raise WorkerError("adapter prepared stale or mixed activation revisions")
        self._prepared_activation_revision = activation_revision
        return prepared

    def load_model(self, *, target: Mapping[str, Any], model_revision: str,
                   tokenizer_revision: str, device: str) -> Any:
        if target != self.contract["target"] or device != self.device:
            raise WorkerError("production adapter target/device differs from sealed contract")
        from wisent.core.primitives.models.wisent_model import WisentModel
        return WisentModel(
            target["model_name"], device=device, revision=model_revision,
            tokenizer_revision=tokenizer_revision,
        )

    def model_identity(self, model: Any) -> dict[str, str]:
        return {
            "model_revision": getattr(model, "resolved_model_revision", None),
            "tokenizer_revision": getattr(model, "resolved_tokenizer_revision", None),
        }

    @staticmethod
    def _pair_map(pair_texts: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
        pairs = _pair_rows(pair_texts)
        result: dict[int, Mapping[str, Any]] = {}
        for row in pairs:
            if not isinstance(row, Mapping) or type(row.get("pair_id")) is not int:
                raise WorkerError("pair_texts contains malformed production rows")
            if row["pair_id"] in result:
                raise WorkerError("pair_texts contains duplicate production pair IDs")
            result[row["pair_id"]] = row
        return result

    @staticmethod
    def _response(row: Mapping[str, Any], field: str) -> str:
        response = row.get(field)
        if not isinstance(response, Mapping) or not isinstance(response.get("model_response"), str):
            raise WorkerError(f"pair_texts row has malformed {field}")
        return response["model_response"]

    @classmethod
    def _write_pairs(cls, path: Path, task: str, rows: Sequence[Mapping[str, Any]]) -> None:
        payload = {
            "task_name": task, "num_pairs": len(rows),
            "pair_ids": [row["pair_id"] for row in rows],
            "pairs": [{
                "pair_id": row["pair_id"], "stable_id": row["stable_id"],
                "prompt": row["prompt"],
                "positive_response": {"model_response": cls._response(row, "positive_response")},
                "negative_response": {"model_response": cls._response(row, "negative_response")},
            } for row in rows],
        }
        path.write_bytes(execution.canonical_json(payload))

    def load_support(self, *, arm: str, pair_texts: Mapping[str, Any],
                     route_payloads: Sequence[Mapping[str, Any]],
                     test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        arm_root = self.root / arm
        arm_root.mkdir()
        by_id = self._pair_map(pair_texts)
        target_support = self.contract["target_manifest"]["support"]["splits"]
        train = []
        for identity in target_support["train"]:
            row = by_id.get(identity["pair_id"])
            if row is None or row.get("stable_id") != identity["stable_id"]:
                raise WorkerError("pair_texts does not preserve sealed train identity")
            train.append(row)
        exact_test = []
        for identity in test_rows:
            row = by_id.get(identity["pair_id"])
            if row is None or row.get("stable_id") != identity["stable_id"]:
                raise WorkerError("pair_texts does not preserve sealed test identity")
            exact_test.append(row)
        test_path = arm_root / "test-pairs.json"
        self._write_pairs(test_path, self.contract["target"]["benchmark"], exact_test)
        return {"root": arm_root, "train_rows": train, "test_path": test_path,
                "routes": list(route_payloads)}

    def install(self, *, model: Any, arm: str, selected_config: Mapping[str, Any] | None,
                selected_artifact: Any, support: Mapping[str, Any]) -> None:
        del model
        if arm == "baseline":
            if selected_config is not None or selected_artifact is not None or support["routes"]:
                raise WorkerError("baseline received steering support")
        else:
            if not isinstance(selected_config, Mapping):
                raise WorkerError("method arm lacks an exact selected config envelope")
            selected_routes = set(execution.selected_config_route_keys(selected_config))
            route_keys = [(route["strategy"], route["layer"]) for route in support["routes"]]
            if len(route_keys) != len(selected_routes) or set(route_keys) != selected_routes:
                raise WorkerError("selected config and support routes differ")
        self._installed[arm] = (selected_config, support)

    @staticmethod
    def _evaluation_args(input_file: str, output_file: str, task: str, model: Any) -> Any:
        from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import _make_args
        from wisent.core.utils.config_tools.constants import (
            EVAL_F1_THRESHOLD, EVAL_GENERATION_EMBEDDING_WEIGHT,
            EVAL_GENERATION_NLI_WEIGHT, SCORE_MIDPOINT_PCT, SPLIT_RATIO_TRAIN_DEFAULT,
        )
        from wisent.core.utils.infra_tools.infra.core.hardware import subprocess_timeout_s
        return _make_args(
            input=input_file, output=output_file, task=task, verbose=False,
            f1_threshold=EVAL_F1_THRESHOLD,
            generation_embedding_weight=EVAL_GENERATION_EMBEDDING_WEIGHT,
            generation_nli_weight=EVAL_GENERATION_NLI_WEIGHT,
            train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
            subprocess_timeout=subprocess_timeout_s(),
            personalization_good_threshold=SCORE_MIDPOINT_PCT, cached_model=model,
        )

    def _baseline(self, model: Any, support: Mapping[str, Any], task: str) -> Mapping[str, Any]:
        from wisent.core.utils.cli.commands.optimize_steering.pipeline.scores import (
            execute_evaluate_responses, task_uses_log_likelihoods, write_placeholder_responses,
        )
        if not task_uses_log_likelihoods(task):
            raise WorkerError("sealed baseline evaluator is not log_likelihoods")
        responses = support["root"] / "responses.json"
        scores = support["root"] / "scores.json"
        write_placeholder_responses(str(support["test_path"]), str(responses),
                                    len(self.contract["target_manifest"]["support"]["splits"]["test"]),
                                    task, self.contract["target"]["model_name"])
        with model.detached():
            execute_evaluate_responses(self._evaluation_args(str(responses), str(scores), task, model))
        return json.loads(scores.read_text())

    def _steered(self, model: Any, arm: str, selected: Mapping[str, Any],
                 support: Mapping[str, Any], task: str) -> Mapping[str, Any]:
        from wisent.core.reading.modules.utilities.data.enriched_builder import build_enriched_from_local_strict
        from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import (
            _build_config, _merge_strict_enriched_inputs, run_pipeline,
        )
        selected_routes = execution.selected_config_route_keys(selected)
        route_keys = [(route["strategy"], route["layer"]) for route in support["routes"]]
        if len(route_keys) != len(selected_routes) or set(route_keys) != set(selected_routes):
            raise WorkerError("selected config and materialized support routes differ")
        strict_enriched: dict[tuple[str, int], str] = {}
        for index, route in enumerate(support["routes"]):
            activation_path = route.get("activation_path")
            proof = route["proof"]
            if not isinstance(activation_path, Path) or not activation_path.is_file():
                raise WorkerError("selected support route lacks its exact local activation")
            route_root = support["root"] / f"route-{index}-{route['layer']}"
            route_root.mkdir()
            strict_enriched[(route["strategy"], route["layer"])] = build_enriched_from_local_strict(
                self.contract["target"]["model_name"], task, route["layer"], route["strategy"],
                str(route_root), [row["pair_id"] for row in support["train_rows"]],
                activation_file=str(activation_path), activation_pair_ids=proof["pair_ids"],
                pair_rows=support["train_rows"],
            )
        params = dict(selected["params"])
        config, strength = _build_config(arm, params)
        enriched = _merge_strict_enriched_inputs(
            strict_enriched, selected["strategy"],
            [layer for _, layer in selected_routes], str(support["root"]),
        )
        pipeline_root = support["root"] / "pipeline"
        pipeline_root.mkdir()
        result = run_pipeline(
            self.contract["target"]["model_name"], task, config, str(pipeline_root), strength,
            device=self.device, enriched_pairs_file=enriched,
            test_pairs_file=str(support["test_path"]),
            evaluation_pairs_file=str(support["test_path"]), cached_model=model,
        )
        return result.details

    @staticmethod
    def _normalize(details: Mapping[str, Any], test_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        evaluations = details.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(test_rows):
            raise WorkerError("production evaluator did not return the exact test batch")
        predictions = []
        correct = 0
        for identity, row in zip(test_rows, evaluations):
            outcome = row.get("evaluation") if isinstance(row, Mapping) else None
            if not isinstance(outcome, Mapping) or type(outcome.get("correct")) is not bool:
                raise WorkerError("production evaluator returned a malformed prediction")
            predictions.append({"pair_id": identity["pair_id"],
                                "stable_id": identity["stable_id"],
                                "correct": outcome["correct"]})
            correct += int(outcome["correct"])
        accuracy = correct / len(test_rows)
        aggregate = details.get("aggregated_metrics", {})
        adjusted = aggregate.get("acc", aggregate.get("overall", accuracy))
        if isinstance(adjusted, bool) or not isinstance(adjusted, (int, float)) or not math.isfinite(adjusted):
            raise WorkerError("production evaluator accuracy is not finite")
        coherence = float(adjusted) / accuracy if accuracy else 1.0
        return {"accuracy": float(adjusted), "coherence": coherence,
                "predictions": predictions}

    def score_batch(
        self, *, model: Any, arm: str, support: Mapping[str, Any],
        test_rows: Sequence[Mapping[str, Any]], evaluator: Mapping[str, Any],
        activation_revision: str,
    ) -> dict[str, Any]:
        if arm not in self._installed or self._installed[arm][1] is not support:
            raise WorkerError("score_batch called outside the installed adapter lifecycle")
        if evaluator["name"] != "log_likelihoods":
            raise WorkerError("production adapter requires log_likelihoods evaluator")
        if (activation_revision != self._prepared_activation_revision or
                activation_revision != self.contract["revisions"]["activation"]):
            raise WorkerError("adapter evaluation activation revision differs from preparation")
        task = self.contract["target"]["benchmark"]
        selected = self._installed[arm][0]
        details = self._baseline(model, support, task) if arm == "baseline" else self._steered(model, arm, selected, support, task)
        observed_runtime = execution.observe_runtime_evidence()
        identity = self.model_identity(model)
        return {
            "scores": self._normalize(details, test_rows),
            "runtime_evidence": {
                "model_revision": identity["model_revision"],
                "tokenizer_revision": identity["tokenizer_revision"],
                "activation_revision": self._prepared_activation_revision,
                "runtime_revision": observed_runtime["runtime_revision"],
                "device": observed_runtime["device"],
            },
        }

    def remove(self, *, model: Any, arm: str) -> None:
        self._installed.pop(arm, None)
        clear = getattr(model, "clear_steering", None)
        if callable(clear):
            clear()

    def close(self, model: Any) -> None:
        try:
            detach = getattr(model, "detach", None)
            if callable(detach):
                detach()
        finally:
            shutil.rmtree(self.root, ignore_errors=True)


def _ref_from_cli(
    store: Any, uri: str, generation: str, option: str,
) -> dict[str, str]:
    if not uri.startswith("gs://"):
        raise WorkerError(f"{option} must be an immutable production gs:// URI")
    if not generation.isdigit() or generation.startswith("0"):
        raise WorkerError(f"{option}-generation must be a canonical positive decimal")
    data, observed = store.read(uri, generation)
    observed_generation = observed.get("generation") if isinstance(observed, Mapping) else observed
    if str(observed_generation) != generation:
        raise WorkerError(f"{option} generation differs from requested generation")
    return execution.artifact_ref(
        uri, generation, str(len(data)), hashlib.sha256(data).hexdigest(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="desired-results-final-test-worker-v3")
    parser.add_argument("--seal-ref", required=True)
    parser.add_argument("--seal-ref-generation", required=True)
    parser.add_argument("--arm-manifest", required=True)
    parser.add_argument("--arm-manifest-generation", required=True)
    parser.add_argument("--attempt-number", type=int, required=True)
    parser.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        store = GCSStore()
        seal_ref = _ref_from_cli(
            store, args.seal_ref, args.seal_ref_generation, "--seal-ref",
        )
        manifest_ref = _ref_from_cli(
            store, args.arm_manifest, args.arm_manifest_generation, "--arm-manifest",
        )
        seal, contract, manifests, _ = _load_sealed_wave(store, seal_ref)
        _select_sealed_arm(seal, manifests, manifest_ref)
        adapter = ProductionFinalTestAdapter(contract, args.device)
        result = execute_arm(
            store, seal_ref, manifest_ref, adapter, args.device,
            attempt=args.attempt_number,
        )
        print(execution.canonical_json(result).decode("ascii"))
        return 0
    except (WorkerError, FinalTestError, execution.ContractError,
            OSError, ValueError, KeyError, TypeError) as exc:
        print(f"desired-results final-test worker refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
