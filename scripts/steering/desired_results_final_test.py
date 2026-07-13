#!/usr/bin/env python3
"""Schema-v3 final-test control plane with immutable create-only publication."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Mapping, Sequence

from scripts.steering import desired_results_execution_contract as execution
from scripts.steering import desired_results_target

SCHEMA_VERSION = execution.SCHEMA_VERSION


class FinalTestError(RuntimeError):
    """The final-test graph or lifecycle is unsafe or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return execution.canonical_json(value)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json(data: bytes, label: str) -> Any:
    try:
        value = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalTestError(f"{label} is not strict ASCII JSON: {exc}") from exc
    if _canonical_bytes(value) != data:
        raise FinalTestError(f"{label} is not canonical JSON")
    return value


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise FinalTestError(f"{label} keys must be exactly {sorted(keys)}; got {actual}")
    return value


def _normalize_read(result: Any, expected_generation: str | None) -> tuple[bytes, str]:
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], bytes):
        raise FinalTestError("store.read must return (bytes, generation)")
    generation = result[1].get("generation") if isinstance(result[1], Mapping) else result[1]
    if generation is None or (expected_generation is not None and str(generation) != expected_generation):
        raise FinalTestError("store returned a different object generation")
    return result[0], str(generation)


def read_ref(store: Any, value: Mapping[str, Any], label: str = "artifact") -> tuple[Any, dict[str, str]]:
    try:
        ref = execution.validate_artifact_ref(value, label)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    data, generation = _normalize_read(store.read(ref["uri"], ref["generation"]), ref["generation"])
    digest = hashlib.sha256(data).hexdigest()
    if str(len(data)) != ref["size"] or digest != ref["sha256"]:
        raise FinalTestError(f"{label} bytes differ from their immutable reference")
    return _strict_json(data, label), execution.artifact_ref(ref["uri"], generation, str(len(data)), digest)


def _published_ref(uri: str, data: bytes, result: Any) -> dict[str, str]:
    generation = result.get("generation") if isinstance(result, Mapping) else result
    if generation is None:
        raise FinalTestError("store did not return an object generation")
    return execution.artifact_ref(uri, str(generation), str(len(data)), hashlib.sha256(data).hexdigest())


def publish_bytes(store: Any, uri: str, data: bytes) -> dict[str, str]:
    """Create once; an existing object is accepted only when its bytes are identical."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise FinalTestError("publication URI must be a production gs:// URI")
    try:
        return _published_ref(uri, data, store.create(uri, data))
    except Exception as create_error:
        try:
            existing, generation = _normalize_read(store.read(uri, None), None)
        except Exception:
            raise FinalTestError(f"create-only publication failed for {uri}: {create_error}") from create_error
        if existing != data:
            raise FinalTestError(f"conflicting object already exists at {uri}") from create_error
        return _published_ref(uri, data, generation)


def create_bytes_once(store: Any, uri: str, data: bytes) -> tuple[dict[str, str], bool]:
    """Attempt one create-only write and report whether this caller acquired it."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise FinalTestError("publication URI must be a production gs:// URI")
    try:
        return _published_ref(uri, data, store.create(uri, data)), True
    except Exception as create_error:
        try:
            existing, generation = _normalize_read(store.read(uri, None), None)
        except Exception:
            raise FinalTestError(f"create-only publication failed for {uri}: {create_error}") from create_error
        if existing != data:
            raise FinalTestError(f"conflicting object already exists at {uri}") from create_error
        return _published_ref(uri, existing, generation), False


def create_json_once(store: Any, uri: str, value: Any) -> tuple[dict[str, str], bool]:
    return create_bytes_once(store, uri, _canonical_bytes(value))


STAGED_RESULT_KEYS = {
    "schema_version", "contract_sha256", "arm_manifest_sha256", "arm", "target",
    "revisions", "runtime_evidence", "runtime_evidence_sha256", "pair_texts_ref",
    "evaluator_ref", "support_refs", "test_token_id", "test_token_consumptions",
    "test_pair_count", "scores",
}


def validate_staged_result(
    value: Any, contract: Mapping[str, Any], manifest: Mapping[str, Any],
    label: str = "staged result",
) -> dict[str, Any]:
    """Validate a worker result against its exact sealed target and arm lineage."""
    result = _exact(value, STAGED_RESULT_KEYS, label)
    try:
        execution.validate_execution_contract(contract)
        execution.validate_arm_manifest(manifest, contract)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "arm_manifest_sha256": manifest["manifest_sha256"],
        "arm": manifest["arm"],
        "target": contract["target"],
        "revisions": contract["revisions"],
        "runtime_evidence": contract["runtime_evidence"],
        "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
        "pair_texts_ref": manifest["pair_texts_ref"],
        "evaluator_ref": manifest["evaluator_ref"],
        "support_refs": manifest["support_refs"],
        "test_token_id": execution.test_token_id(manifest["manifest_sha256"]),
        "test_token_consumptions": 1,
        "test_pair_count": len(contract["target_manifest"]["support"]["splits"]["test"]),
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise FinalTestError(f"{label}.{field} differs from sealed lineage")
    scores = result["scores"]
    if not isinstance(scores, Mapping):
        raise FinalTestError(f"{label}.scores must be an object")
    required = contract["target_manifest"]["evaluation"]["required_outputs"]
    missing = [name for name in required if name not in scores]
    if missing:
        raise FinalTestError(f"{label}.scores misses required outputs: {missing}")
    predictions = scores.get("predictions")
    test_rows = contract["target_manifest"]["support"]["splits"]["test"]
    if not isinstance(predictions, list) or len(predictions) != len(test_rows):
        raise FinalTestError(f"{label}.predictions differs from held-out batch size")
    for expected_row, prediction in zip(test_rows, predictions, strict=True):
        if (not isinstance(prediction, Mapping)
                or prediction.get("pair_id") != expected_row["pair_id"]
                or prediction.get("stable_id") != expected_row["stable_id"]):
            raise FinalTestError(f"{label}.predictions differs from ordered pair identity")
    _canonical_bytes(result)
    return dict(result)


def read_completion_lineage(
    store: Any, completion_ref: Mapping[str, Any], contract: Mapping[str, Any],
    manifest: Mapping[str, Any], label: str = "completion",
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    """Dereference and exact-validate completion, attempt, staged and published bytes."""
    completion, normalized_completion_ref = read_ref(store, completion_ref, label)
    if not isinstance(completion, Mapping):
        raise FinalTestError(f"{label} must be an object")
    try:
        execution.validate_completion_receipt(completion, contract, manifest)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    expected_completion_uri = manifest["output_prefix"] + "completion.json"
    if normalized_completion_ref["uri"] != expected_completion_uri:
        raise FinalTestError(f"{label} ref is foreign to the sealed arm output prefix")
    attempt_ref_value = completion["attempt_receipt_ref"]
    attempt, normalized_attempt_ref = read_ref(store, attempt_ref_value, f"{label} attempt")
    try:
        execution.validate_completion_receipt(completion, contract, manifest, attempt)
        execution.validate_artifact_binding(normalized_completion_ref, completion, f"{label} ref")
        expected_attempt_ref = execution.validate_artifact_ref(completion["attempt_receipt_ref"])
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    if normalized_attempt_ref != expected_attempt_ref:
        raise FinalTestError(f"{label} attempt ref changed")
    expected_attempt_uri = (
        manifest["output_prefix"] + f"attempts/{attempt['attempt']}/completed.json"
    )
    expected_stage_uri = (
        manifest["output_prefix"] + f"attempts/{attempt['attempt']}/staged-result.json"
    )
    expected_publication_uri = manifest["output_prefix"] + "result.json"
    if normalized_attempt_ref["uri"] != expected_attempt_uri:
        raise FinalTestError(f"{label} attempt ref is foreign to its arm/attempt lineage")
    if completion["staged_result_ref"]["uri"] != expected_stage_uri:
        raise FinalTestError(f"{label} staged result ref is foreign to its arm/attempt lineage")
    if completion["publication_ref"]["uri"] != expected_publication_uri:
        raise FinalTestError(f"{label} publication ref is foreign to its sealed arm")
    staged, normalized_staged_ref = read_ref(
        store, completion["staged_result_ref"], f"{label} staged result",
    )
    published, normalized_publication_ref = read_ref(
        store, completion["publication_ref"], f"{label} publication",
    )
    try:
        expected_staged_ref = execution.validate_artifact_ref(completion["staged_result_ref"])
        expected_publication_ref = execution.validate_artifact_ref(completion["publication_ref"])
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    if normalized_staged_ref != expected_staged_ref:
        raise FinalTestError(f"{label} staged result ref changed")
    if normalized_publication_ref != expected_publication_ref:
        raise FinalTestError(f"{label} publication ref changed")
    staged = validate_staged_result(staged, contract, manifest, f"{label} staged result")
    published = validate_staged_result(published, contract, manifest, f"{label} publication")
    if staged != published:
        raise FinalTestError(f"{label} publication differs from staged result")
    return dict(completion), normalized_completion_ref, dict(attempt), staged


def publish_json(store: Any, uri: str, value: Any) -> dict[str, str]:
    return publish_bytes(store, uri, _canonical_bytes(value))


def _validate_evaluator(value: Any, ref: Any) -> tuple[dict[str, Any], dict[str, str]]:
    evaluator = _exact(value, {"name", "version", "options"}, "evaluator")
    if not all(isinstance(evaluator[key], str) and evaluator[key] for key in ("name", "version")):
        raise FinalTestError("evaluator name/version must be non-empty strings")
    if not isinstance(evaluator["options"], Mapping):
        raise FinalTestError("evaluator options must be an object")
    normalized = {"name": evaluator["name"], "version": evaluator["version"], "options": dict(evaluator["options"])}
    try:
        execution.validate_artifact_binding(ref, normalized, "evaluator_ref")
        normalized_ref = execution.validate_artifact_ref(ref, "evaluator_ref")
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    return normalized, normalized_ref


def _validate_prepared_target(value: Any, evaluator: Mapping[str, Any]) -> dict[str, Any]:
    item = _exact(value, {"target_manifest", "target_manifest_ref", "calibrations"}, "prepared target")
    manifest = item["target_manifest"]
    try:
        desired_results_target.validate_target_manifest(manifest)
        execution.validate_artifact_binding(item["target_manifest_ref"], manifest, "target_manifest_ref")
        manifest_ref = execution.validate_artifact_ref(item["target_manifest_ref"], "target_manifest_ref")
    except (desired_results_target.ContractError, execution.ContractError) as exc:
        raise FinalTestError(str(exc)) from exc
    lifecycle = manifest["execution"]
    if (manifest["activation"]["eligible"] is not True or
            lifecycle["state"] != "unprepared" or lifecycle["blocked"] is not False or
            lifecycle["rerun_locked"] is not False):
        raise FinalTestError(
            "final-test preparation requires an activation-eligible, unprepared, "
            "unblocked, rerun-unlocked target"
        )
    if manifest["support"]["state"] != "prepared" or manifest["evaluation"]["split"] != "test":
        raise FinalTestError("final test requires prepared support and held-out test evaluation")
    methods = manifest["calibration"]["methods"]
    raw_calibrations = item["calibrations"]
    if not isinstance(raw_calibrations, Sequence) or isinstance(raw_calibrations, (str, bytes)):
        raise FinalTestError("calibrations must be a sequence")
    by_method: dict[str, dict[str, Any]] = {}
    expected_target = {key: manifest["target"][key] for key in ("target_id", "model_name", "model_slug", "benchmark")}
    for index, raw in enumerate(raw_calibrations):
        calibration = _exact(raw, {"manifest", "receipt"}, f"calibrations[{index}]")
        calibration_manifest, receipt = calibration["manifest"], calibration["receipt"]
        try:
            execution.validate_calibration_manifest(calibration_manifest, manifest)
            execution.validate_calibration_success_receipt(receipt, calibration_manifest)
        except execution.ContractError as exc:
            raise FinalTestError(str(exc)) from exc
        method = calibration_manifest["method"]
        if method in by_method:
            raise FinalTestError(f"duplicate successful calibration for {method}")
        if calibration_manifest["target"] != expected_target or calibration_manifest["evaluator"] != evaluator:
            raise FinalTestError("calibration target/evaluator differs from final target/evaluator")
        if receipt["selected_config"].get("method") != method:
            raise FinalTestError("successful calibration selected_config method differs")
        by_method[method] = {"manifest": calibration_manifest, "receipt": receipt}
    if set(by_method) != set(methods):
        raise FinalTestError(f"calibrations must cover target methods exactly; expected={methods}, got={sorted(by_method)}")
    return {"target_manifest": manifest, "target_manifest_ref": manifest_ref,
            "calibrations": [by_method[method] for method in methods]}


def plan(
    prepared_targets: Sequence[Mapping[str, Any]],
    evaluator: Mapping[str, Any],
    evaluator_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a nested composite plan without arm intersection or silent omission."""
    evaluator, evaluator_ref = _validate_evaluator(evaluator, evaluator_ref)
    if not isinstance(prepared_targets, Sequence) or isinstance(prepared_targets, (str, bytes)) or not prepared_targets:
        raise FinalTestError("plan requires at least one prepared target")
    targets = [_validate_prepared_target(item, evaluator) for item in prepared_targets]
    target_ids = [item["target_manifest"]["target"]["target_id"] for item in targets]
    if len(target_ids) != len(set(target_ids)):
        raise FinalTestError("composite plan contains duplicate targets")
    return {"schema_version": SCHEMA_VERSION, "evaluator": evaluator,
            "evaluator_ref": evaluator_ref, "targets": targets}


def prepare(
    planned: Mapping[str, Any], *, protocol: Mapping[str, Any],
    revisions_by_target: Mapping[str, Mapping[str, Any]],
    calibration_policy_by_target: Mapping[str, Mapping[str, Any]], output_namespace: str,
    runtime_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build one centrally validated ExecutionContractV3 per target."""
    root = _exact(planned, {"schema_version", "evaluator", "evaluator_ref", "targets"}, "plan")
    if root["schema_version"] != SCHEMA_VERSION:
        raise FinalTestError("plan schema_version must be 3")
    evaluator, evaluator_ref = _validate_evaluator(root["evaluator"], root["evaluator_ref"])
    try:
        runtime = execution.validate_runtime_evidence(runtime_evidence)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    contracts: list[dict[str, Any]] = []
    for raw_target in root["targets"]:
        item = _validate_prepared_target(raw_target, evaluator)
        manifest = item["target_manifest"]
        target_id = manifest["target"]["target_id"]
        if target_id not in revisions_by_target:
            raise FinalTestError(f"missing exact revisions for {target_id}")
        if target_id not in calibration_policy_by_target:
            raise FinalTestError(f"missing calibration policy for {target_id}")
        methods = list(manifest["calibration"]["methods"])
        manifest_policies = {
            entry["manifest"]["method"]: entry["manifest"]["calibration_policy"]
            for entry in item["calibrations"]
        }
        first_policy = manifest_policies[methods[0]]
        first_options = first_policy["options"]
        first_optimizer = first_options["optimizer"]
        for method in methods:
            candidate = manifest_policies[method]
            optimizer = candidate["options"]["optimizer"]
            if ({key: candidate[key] for key in ("name", "version", "policy_ref")} !=
                    {key: first_policy[key] for key in ("name", "version", "policy_ref")} or
                    candidate["options"]["device"] != first_options["device"] or
                    {key: optimizer[key] for key in ("backend", "direction", "seed")} !=
                    {key: first_optimizer[key] for key in ("backend", "direction", "seed")}):
                raise FinalTestError(f"calibration policy identity differs across methods for {target_id}")
        expected_policy = {
            "name": first_policy["name"], "version": first_policy["version"],
            "policy_ref": first_policy["policy_ref"],
            "options": {"device": first_options["device"], "optimizer": {
                "backend": first_optimizer["backend"], "direction": first_optimizer["direction"],
                "seed": first_optimizer["seed"],
                "trials_per_strategy": {method: manifest_policies[method]["options"]["optimizer"]["trials_per_strategy"] for method in methods},
                "method_space": {method: manifest_policies[method]["options"]["optimizer"]["method_space"] for method in methods},
            }},
        }
        calibration_policy = calibration_policy_by_target[target_id]
        if calibration_policy != expected_policy:
            raise FinalTestError(f"aggregate calibration policy differs from prepared method policies for {target_id}")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol": dict(protocol),
            "target_manifest": manifest,
            "target_manifest_ref": item["target_manifest_ref"],
            "target": {key: manifest["target"][key] for key in ("target_id", "model_name", "model_slug", "benchmark")},
            "revisions": dict(revisions_by_target[target_id]),
            "matrix": {"strategies": list(desired_results_target.STRATEGIES),
                       "layers": list(range(1, manifest["calibration"]["layer_count"] + 1)),
                       "methods": methods, "pairs": manifest["target"]["expected_pairs"],
                       "splits": dict(manifest["support"]["split_counts"])},
            "calibration_policy": dict(calibration_policy),
            "calibration_receipts": [entry["receipt"] for entry in item["calibrations"]],
            "evaluator": evaluator,
            "evaluator_ref": evaluator_ref,
            "final_test": {"split": "test", "evaluations_per_arm": 1},
            "arms": ["baseline", *methods],
            "retry_policy": {"max_pre_test_attempts": execution.MAX_PRE_TEST_ATTEMPTS},
            "output_namespace": output_namespace.rstrip("/") + "/" + manifest["target"]["model_slug"] + "/" + manifest["target"]["benchmark"],
            "runtime_evidence": runtime,
            "runtime_evidence_sha256": execution.runtime_evidence_sha256(runtime),
        }
        try:
            contracts.append(execution.finalize_execution_contract(payload))
        except execution.ContractError as exc:
            raise FinalTestError(str(exc)) from exc
    return contracts


def arm_manifests(contract: Mapping[str, Any], contract_ref: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive every ArmManifestV3 exclusively from the sealed contract."""
    try:
        execution.validate_execution_contract(contract)
        execution.validate_artifact_binding(contract_ref, contract, "contract_ref")
        contract_ref = execution.validate_artifact_ref(contract_ref, "contract_ref")
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    receipts = {receipt["selected_config"]["method"]: receipt for receipt in contract["calibration_receipts"]}
    result: dict[str, dict[str, Any]] = {}
    for arm in contract["arms"]:
        receipt = None if arm == "baseline" else receipts[arm]
        support_refs = []
        if receipt is not None:
            selected_keys = set(execution.selected_config_route_keys(receipt["selected_config"]))
            support_refs = [
                {key: route[key] for key in ("strategy", "layer", "completion_ref", "proof_ref")}
                for route in contract["target_manifest"]["activation"]["routes"]
                if (route["strategy"], route["layer"]) in selected_keys
            ]
        payload = {
            "schema_version": SCHEMA_VERSION, "contract_ref": contract_ref,
            "contract_sha256": contract["contract_sha256"], "arm": arm,
            "method": None if arm == "baseline" else arm,
            "selected_config_ref": None if receipt is None else receipt["result_ref"],
            "pair_texts_ref": contract["target_manifest"]["support"]["pair_texts_ref"],
            "support_refs": support_refs, "evaluator_ref": contract["evaluator_ref"],
            "runtime_evidence": contract["runtime_evidence"],
            "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
            "output_namespace": contract["output_namespace"],
            "output_prefix": execution.derive_output_prefix(contract["output_namespace"], contract["contract_sha256"], arm),
        }
        try:
            result[arm] = execution.finalize_arm_manifest(payload)
        except execution.ContractError as exc:
            raise FinalTestError(str(exc)) from exc
    return result


def seal_target(store: Any, contract: Mapping[str, Any], *, control_prefix: str) -> dict[str, Any]:
    """Publish contract, all arms, then the seal and one target-local wave."""
    if not isinstance(control_prefix, str) or not control_prefix.startswith("gs://"):
        raise FinalTestError("control_prefix must be gs://")
    target_key = hashlib.sha256(contract["target"]["target_id"].encode("utf-8")).hexdigest()[:16]
    target_prefix = f"{control_prefix.rstrip('/')}/targets/{target_key}"
    contract_ref = publish_json(
        store, f"{target_prefix}/contracts/{contract['contract_sha256']}.json", contract,
    )
    manifests = arm_manifests(contract, contract_ref)
    manifest_refs = {
        arm: publish_json(
            store, f"{target_prefix}/arms/{arm}/{manifest['manifest_sha256']}.json", manifest,
        )
        for arm, manifest in manifests.items()
    }
    try:
        seal = execution.finalize_final_seal({
            "schema_version": SCHEMA_VERSION, "contract": contract, "contract_ref": contract_ref,
            "contract_sha256": contract["contract_sha256"], "arm_manifest_refs": manifest_refs,
            "runtime_evidence": contract["runtime_evidence"],
            "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
        })
        execution.validate_final_seal(seal, manifests)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    seal_ref = publish_json(
        store, f"{target_prefix}/seals/{seal['seal_sha256']}.json", seal,
    )
    wave = {"target_id": contract["target"]["target_id"], "contract_ref": contract_ref,
            "seal_ref": seal_ref, "arm_manifest_refs": manifest_refs, "arms": list(contract["arms"])}
    return {"contract": contract, "contract_ref": contract_ref, "manifests": manifests,
            "manifest_refs": manifest_refs, "seal": seal, "seal_ref": seal_ref, "wave": wave}


def waves(sealed_targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one wave per target, never intersecting arm sets across targets."""
    result: list[dict[str, Any]] = []
    for item in sealed_targets:
        wave = item.get("wave") if isinstance(item, Mapping) else None
        if not isinstance(wave, Mapping) or set(wave.get("arms", ())) != set(wave.get("arm_manifest_refs", {})):
            raise FinalTestError("sealed wave does not cover its target arms exactly")
        for ref in (wave["contract_ref"], wave["seal_ref"], *wave["arm_manifest_refs"].values()):
            if not execution.validate_artifact_ref(ref)["uri"].startswith("gs://"):
                raise FinalTestError("local refs cannot be dispatched")
        result.append(dict(wave))
    return result


def _read_sealed_graph(
    store: Any, sealed_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    sealed = _exact(
        sealed_value,
        {"contract", "contract_ref", "manifests", "manifest_refs", "seal", "seal_ref", "wave"},
        "sealed target",
    )
    seal, seal_ref = read_ref(store, sealed["seal_ref"], "final seal")
    try:
        execution.validate_final_seal(seal)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    if seal != sealed["seal"]:
        raise FinalTestError("sealed target final seal differs from referenced bytes")
    contract, contract_ref = read_ref(store, seal["contract_ref"], "execution contract")
    if contract != seal["contract"] or contract != sealed["contract"]:
        raise FinalTestError("sealed target execution contract differs from referenced bytes")
    try:
        sealed_contract_ref = execution.validate_artifact_ref(sealed["contract_ref"])
        expected_contract_ref = execution.validate_artifact_ref(seal["contract_ref"])
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    if contract_ref != expected_contract_ref:
        raise FinalTestError("sealed target execution contract ref changed")
    if contract_ref != sealed_contract_ref:
        raise FinalTestError("sealed target carries a foreign execution contract ref")
    seal_uri = seal_ref["uri"]
    seal_suffix = f"/seals/{seal['seal_sha256']}.json"
    if not seal_uri.startswith("gs://") or not seal_uri.endswith(seal_suffix):
        raise FinalTestError("final seal ref is foreign to its canonical target namespace")
    target_prefix = seal_uri[:-len(seal_suffix)]
    expected_contract_uri = f"{target_prefix}/contracts/{contract['contract_sha256']}.json"
    if contract_ref["uri"] != expected_contract_uri:
        raise FinalTestError("execution contract ref is foreign to the final seal target namespace")
    raw_manifests = sealed["manifests"]
    raw_manifest_refs = sealed["manifest_refs"]
    if not isinstance(raw_manifests, Mapping) or not isinstance(raw_manifest_refs, Mapping):
        raise FinalTestError("sealed target manifests and refs must be arm maps")
    if set(raw_manifests) != set(contract["arms"]) or set(raw_manifest_refs) != set(contract["arms"]):
        raise FinalTestError("sealed target manifest maps do not cover every arm exactly")
    manifests: dict[str, dict[str, Any]] = {}
    for arm in contract["arms"]:
        manifest, manifest_ref = read_ref(store, seal["arm_manifest_refs"][arm], f"arm manifest {arm}")
        if manifest != raw_manifests[arm]:
            raise FinalTestError(f"sealed target arm manifest differs from referenced bytes for {arm}")
        try:
            expected_manifest_ref = execution.validate_artifact_ref(seal["arm_manifest_refs"][arm])
            supplied_manifest_ref = execution.validate_artifact_ref(raw_manifest_refs[arm])
        except execution.ContractError as exc:
            raise FinalTestError(str(exc)) from exc
        if manifest_ref != expected_manifest_ref:
            raise FinalTestError(f"sealed target arm manifest ref changed for {arm}")
        if manifest_ref != supplied_manifest_ref:
            raise FinalTestError(f"sealed target carries a foreign arm manifest ref for {arm}")
        expected_manifest_uri = f"{target_prefix}/arms/{arm}/{manifest['manifest_sha256']}.json"
        if manifest_ref["uri"] != expected_manifest_uri:
            raise FinalTestError(f"arm manifest ref is foreign to the final seal target namespace for {arm}")
        manifests[arm] = dict(manifest)
    try:
        execution.validate_final_seal(seal, manifests)
        execution.validate_artifact_binding(seal_ref, seal, "final seal ref")
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    wave = _exact(
        sealed["wave"],
        {"target_id", "contract_ref", "seal_ref", "arm_manifest_refs", "arms"},
        "sealed target wave",
    )
    try:
        normalized_manifest_refs = {
            arm: execution.validate_artifact_ref(raw_manifest_refs[arm]) for arm in contract["arms"]
        }
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    expected_wave = {
        "target_id": contract["target"]["target_id"],
        "contract_ref": contract_ref,
        "seal_ref": seal_ref,
        "arm_manifest_refs": normalized_manifest_refs,
        "arms": list(contract["arms"]),
    }
    if wave != expected_wave:
        raise FinalTestError("sealed target wave differs from dereferenced control graph")
    return dict(seal), dict(contract), manifests, seal_ref


def finalize_target(
    store: Any, sealed: Mapping[str, Any], completion_refs: Mapping[str, Mapping[str, Any]],
    *, result_uri: str, finalization_uri: str,
) -> dict[str, Any]:
    """Publish a deterministic final result only after exact lineage verification."""
    _, contract, manifests, seal_ref = _read_sealed_graph(store, sealed)
    if not isinstance(completion_refs, Mapping) or set(completion_refs) != set(contract["arms"]):
        raise FinalTestError("completion refs must cover every arm exactly")
    completions: list[dict[str, Any]] = []
    normalized_refs: dict[str, dict[str, str]] = {}
    for arm in contract["arms"]:
        completion, ref, _, _ = read_completion_lineage(
            store, completion_refs[arm], contract, manifests[arm], f"completion[{arm}]",
        )
        if completion["arm"] != arm:
            raise FinalTestError(f"completion arm differs for {arm}")
        completions.append(completion)
        normalized_refs[arm] = ref
    final_result = {"schema_version": SCHEMA_VERSION, "contract_sha256": contract["contract_sha256"],
                    "target": dict(contract["target"]), "evaluator": dict(contract["evaluator"]),
                    "arms": list(contract["arms"]), "completion_refs": normalized_refs}
    result_ref = publish_json(store, result_uri, final_result)
    try:
        execution.validate_artifact_binding(result_ref, final_result, "final result ref")
        receipt = execution.finalize_finalization_receipt({
            "schema_version": SCHEMA_VERSION, "contract_sha256": contract["contract_sha256"],
            "seal_ref": seal_ref, "completion_refs": normalized_refs,
            "final_result_ref": result_ref,
        })
        execution.validate_finalization_receipt(receipt, contract, completions)
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    finalization_ref = publish_json(store, finalization_uri, receipt)
    try:
        execution.validate_artifact_binding(finalization_ref, receipt, "finalization ref")
    except execution.ContractError as exc:
        raise FinalTestError(str(exc)) from exc
    return {"result": final_result, "result_ref": result_ref,
            "finalization": receipt, "finalization_ref": finalization_ref}


class GCSStore:
    """Generation-pinned create-only Google Cloud Storage adapter."""
    def __init__(self) -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise FinalTestError("google-cloud-storage is required") from exc
        self.client = storage.Client()

    def _blob(self, uri: str) -> Any:
        if not uri.startswith("gs://"):
            raise FinalTestError("GCS URI must start with gs://")
        bucket, separator, name = uri[5:].partition("/")
        if not separator or not bucket or not name:
            raise FinalTestError("GCS URI must identify an object")
        return self.client.bucket(bucket).blob(name)

    def exists(self, uri: str) -> bool:
        return bool(self._blob(uri).exists())

    def create(self, uri: str, data: bytes) -> dict[str, str]:
        if not isinstance(data, bytes):
            raise FinalTestError("GCS create data must be bytes")
        blob = self._blob(uri)
        blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
        blob.reload()
        return execution.artifact_ref(
            uri, str(blob.generation), str(len(data)), hashlib.sha256(data).hexdigest(),
        )

    def read(self, uri: str, generation: str | None = None) -> tuple[bytes, str]:
        blob = self._blob(uri)
        if generation is not None:
            blob.generation = int(generation)
        data = blob.download_as_bytes(
            if_generation_match=int(generation) if generation is not None else None,
        )
        blob.reload()
        observed = str(blob.generation)
        if generation is not None and observed != str(generation):
            raise FinalTestError(f"generation drift for {uri}")
        return data, observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="desired-results-final-test-v3")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv); return 0


if __name__ == "__main__":
    raise SystemExit(main())
