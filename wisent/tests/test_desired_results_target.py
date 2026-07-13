import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "steering" / "desired_results_target.py"
SPEC = importlib.util.spec_from_file_location("desired_results_target", PATH)
targets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(targets)

POLICY_PATH = ROOT / "scripts" / "steering" / "desired_results_policy.py"
POLICY_SPEC = importlib.util.spec_from_file_location("desired_results_policy", POLICY_PATH)
policy = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(policy)


def _manifest_payload(
    *,
    model_name="org/model-small",
    model_slug="org__model-small",
    benchmark="short_bench",
    layers=3,
    split_counts=(3, 2, 1),
    activation_status="complete",
    execution_state="unprepared",
    blocked=False,
    rerun_locked=None,
    support_state=None,
    evaluation_split="test",
):
    protocol = "desired-results-v2"
    finalized = execution_state == "finalized"
    locked = finalized if rerun_locked is None else rerun_locked
    pair_count = sum(split_counts)
    split_names = ("train", "validation", "test")
    splits = {}
    cursor = 0
    for name, count in zip(split_names, split_counts, strict=True):
        splits[name] = [
            {"pair_id": index, "stable_id": f"stable-{index}"}
            for index in range(cursor, cursor + count)
        ]
        cursor += count

    def route(strategy, layer):
        identity = f"/{model_slug}/{benchmark}/{strategy}/{layer}/"
        return {
            "strategy": strategy,
            "layer": layer,
            "completion_ref": {
                "uri": f"gs://activation-artifacts{identity}completion.json",
                "generation": "actual-generation-17",
                "size": "128",
                "sha256": "3" * 64,
            },
            "proof_ref": {
                "uri": f"gs://activation-proofs{identity}proof.json",
                "generation": "actual-generation-23",
                "size": "128",
                "sha256": "4" * 64,
            },
        }

    complete_routes = [
        route(strategy, layer)
        for strategy in targets.STRATEGIES
        for layer in range(1, layers + 1)
    ]
    if activation_status == "complete":
        activation = {
            "status": "complete",
            "eligible": execution_state == "unprepared" and not blocked and not locked,
            "layer_count": layers,
            "n_pairs": pair_count,
            "grouped": False,
            "strategies": {name: layers for name in targets.STRATEGIES},
            "routes": complete_routes,
            "proof": {"cache_sha256": "a" * 64, "record_sha256": "b" * 64},
        }
    elif activation_status == "partial":
        activation = {
            "status": "partial",
            "eligible": False,
            "layer_count": layers,
            "n_pairs": pair_count - 1,
            "grouped": False,
            "strategies": {
                name: layers - (name == "mc_balanced") for name in targets.STRATEGIES
            },
            "routes": complete_routes[:-1],
            "proof": {"cache_sha256": "a" * 64, "record_sha256": "b" * 64},
        }
    else:
        activation = {
            "status": "absent",
            "eligible": False,
            "layer_count": None,
            "n_pairs": None,
            "grouped": None,
            "strategies": {name: 0 for name in targets.STRATEGIES},
            "routes": [],
            "proof": {"cache_sha256": "a" * 64, "record_sha256": None},
        }

    prepared = execution_state in {"prepared", "calibrated", "finalized"}
    support_prepared = (
        prepared or activation_status == "complete"
        if support_state is None
        else support_state == "prepared"
    )
    pair_texts_ref = {
        "uri": f"gs://pair-texts/{model_slug}/{benchmark}/pairs.json",
        "generation": "actual-generation-29",
        "size": "128",
        "sha256": "5" * 64,
    }
    support = {
        "state": "prepared" if support_prepared else "missing",
        "proof_sha256": "c" * 64 if support_prepared else None,
        "pair_count": pair_count if support_prepared else 0,
        "split_counts": dict(zip(split_names, split_counts, strict=True))
        if support_prepared
        else {name: 0 for name in split_names},
        "splits": splits if support_prepared else {name: [] for name in split_names},
        "pair_texts_ref": pair_texts_ref if support_prepared else None,
    }
    publication = {
        "uri": f"gs://bucket/results/{model_slug}/{benchmark}/execution-v3/publication.json",
        "generation": "17",
        "size": "608",
        "sha256": "d" * 64,
    } if finalized else None

    return {
        "schema_version": 2,
        "protocol": {"id": protocol, "revision": 1},
        "target": {
            "target_id": targets.target_id(protocol, model_slug, benchmark),
            "result_id": targets.result_id(protocol, model_slug, benchmark),
            "model_name": model_name,
            "model_slug": model_slug,
            "benchmark": benchmark,
            "expected_pairs": pair_count,
            "result_prefix": f"results/{protocol}/{model_slug}/{benchmark}",
        },
        "revisions": {
            "inventory_sha256": "e" * 64,
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "activation_revision": "f" * 40,
        },
        "activation": activation,
        "support": support,
        "evaluation": {"required_outputs": ["accuracy", "coherence"], "split": evaluation_split},
        "calibration": {
            "methods": list(targets.METHODS),
            "strategies": list(targets.STRATEGIES),
            "layer_count": None if activation_status == "absent" else layers,
            "expected_pairs": pair_count,
        },
        "execution": {
            "state": execution_state,
            "blocked": blocked,
            "rerun_locked": locked,
            "publication": publication,
            "provenance": {
                "execution_sha256": "1" * 64 if finalized else None,
                "contract_sha256": "2" * 64 if finalized else None,
            },
        },
    }


def _manifest_binding(manifest):
    encoded = policy.canonical_json(manifest)
    return {
        "ref": {
            "uri": "gs://target-manifests/org__model-small/short_bench/target.json",
            "generation": "actual-generation-31",
            "size": str(len(encoded)),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
        "payload": manifest,
    }


def _policy_inputs(manifest):
    target_id = manifest["target"]["target_id"]
    resource_class = "calibration-gpu"
    return {
        "target_manifest_refs": [_manifest_binding(manifest)],
        "pair_text_refs": {target_id: copy.deepcopy(manifest["support"]["pair_texts_ref"])},
        "revisions": {"code": "3" * 40, "runtime": "3" * 40},
        "resource_classes": {
            resource_class: {
                "accelerator": "nvidia-tesla-a100",
                "memory_bytes": 40_000_000_000,
                "runtime_seconds": 3_600,
                "image": {
                    "name": policy.desired_results_stado.STADO_IMAGE_NAME,
                    "project": policy.desired_results_stado.STADO_IMAGE_PROJECT,
                },
                "dependency_lock_ref": {
                    "uri": "gs://wisent-runtime/requirements.lock",
                    "generation": "actual-generation-37",
                    "size": "128",
                    "sha256": "4" * 64,
                },
            }
        },
        "model_classes": {
            manifest["target"]["model_name"]: {
                "phase_classes": {phase: resource_class for phase in policy.PHASES},
                "hidden_size": 4_096,
            }
        },
        "output_namespace": "gs://wisent-policy/results",
        "trials_per_strategy": 1,
        "retry_policy": {
            "calibration_max_attempts": 1,
            "max_pre_test_attempts": 1,
        },
        "evaluator": {"name": "log_likelihoods", "version": "1", "options": {}},
    }


def test_build_policy_bundle_accepts_activation_ready_unprepared_target():
    manifest = targets.finalize_target_manifest(_manifest_payload())
    inputs = _policy_inputs(manifest)

    bundle = policy.build_policy_bundle(**inputs)

    target_id = manifest["target"]["target_id"]
    assert bundle["target_manifest_refs"] == inputs["target_manifest_refs"]
    assert bundle["pair_text_refs"] == {
        target_id: manifest["support"]["pair_texts_ref"]
    }


@pytest.mark.parametrize(
    "manifest_options",
    [
        pytest.param({"execution_state": "finalized"}, id="finalized"),
        pytest.param({"blocked": True}, id="blocked"),
        pytest.param({"rerun_locked": True}, id="rerun-locked"),
        pytest.param({"activation_status": "partial"}, id="activation-ineligible"),
        pytest.param({"support_state": "missing"}, id="support-missing"),
        pytest.param({"evaluation_split": "validation"}, id="non-test-evaluation"),
    ],
)
def test_build_policy_bundle_rejects_targets_not_ready_for_activation(manifest_options):
    manifest = targets.finalize_target_manifest(_manifest_payload(**manifest_options))

    with pytest.raises(
        policy.PolicyError,
        match="not activation-eligible for calibration/final planning",
    ):
        policy.build_policy_bundle(**_policy_inputs(manifest))


@pytest.mark.parametrize(
    ("model_name", "model_slug", "benchmark", "layers", "split_counts"),
    [
        ("org/model-small", "org__model-small", "short_bench", 3, (3, 2, 1)),
        ("other/model-wide", "other__model-wide", "long_bench", 5, (4, 3, 2)),
    ],
)
def test_manifest_identity_hash_and_route_counts_are_content_derived(
    model_name, model_slug, benchmark, layers, split_counts
):
    payload = _manifest_payload(
        model_name=model_name,
        model_slug=model_slug,
        benchmark=benchmark,
        layers=layers,
        split_counts=split_counts,
    )

    manifest = targets.finalize_target_manifest(payload)
    reordered = json.loads(json.dumps(payload, sort_keys=True))
    reordered_manifest = targets.finalize_target_manifest(reordered)

    assert set(manifest) == targets.TOP_KEYS
    assert manifest["manifest_sha256"] == targets.canonical_sha256(payload)
    assert reordered_manifest["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["target"]["expected_pairs"] == sum(split_counts)
    assert manifest["support"]["split_counts"] == dict(
        zip(("train", "validation", "test"), split_counts, strict=True)
    )
    assert manifest["activation"]["strategies"] == {
        strategy: layers for strategy in targets.STRATEGIES
    }


def test_schema_and_content_hash_reject_unknown_data_and_post_hash_mutation():
    payload = _manifest_payload()
    payload["support"]["prompt"] = "must never leak prompts into a target manifest"
    with pytest.raises(targets.ContractError, match="support keys must be exactly"):
        targets.finalize_target_manifest(payload)

    manifest = targets.finalize_target_manifest(_manifest_payload())
    manifest["target"]["model_name"] = "other/model"
    with pytest.raises(targets.ContractError, match="manifest_sha256"):
        targets.validate_target_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    [
        ("target_id", "desired-results-v2:other__model:short_bench", "target_id/result_id"),
        ("result_id", "desired-results-v2:org__model-small:other_bench", "target_id/result_id"),
        ("result_prefix", "results/desired-results-v2/other__model/short_bench", "result_prefix"),
    ],
)
def test_identity_and_prefix_cannot_cross_target_boundaries(field, bad_value, error):
    payload = _manifest_payload()
    payload["target"][field] = bad_value
    with pytest.raises(targets.ContractError, match=error):
        targets.finalize_target_manifest(payload)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda value: value["support"]["splits"]["test"].__setitem__(0, {"pair_id": 0, "stable_id": "stable-0"}), "globally unique"),
        (lambda value: value["support"]["split_counts"].__setitem__("validation", 1), "split count mismatch"),
        (lambda value: value["support"].__setitem__("pair_count", 5), "pair_count must equal"),
    ],
)
def test_split_partition_rejects_overlap_and_count_drift(mutate, error):
    payload = _manifest_payload()
    mutate(payload)
    with pytest.raises(targets.ContractError, match=error):
        targets.finalize_target_manifest(payload)


@pytest.mark.parametrize(
    ("status", "eligible"),
    [("complete", True), ("partial", False), ("absent", False)],
)
def test_activation_state_controls_eligibility(status, eligible):
    manifest = targets.finalize_target_manifest(
        _manifest_payload(activation_status=status, blocked=status == "absent")
    )
    assert manifest["activation"]["eligible"] is eligible


@pytest.mark.parametrize(
    "mutation",
    [
        lambda activation: activation.__setitem__("n_pairs", activation["n_pairs"] - 1),
        lambda activation: activation["strategies"].__setitem__("role_play", activation["layer_count"] - 1),
        lambda activation: activation["strategies"].__setitem__("chat_first", activation["layer_count"] + 1),
    ],
)
def test_complete_requires_the_exact_strategy_layer_support_matrix(mutation):
    payload = _manifest_payload(layers=5, split_counts=(4, 3, 2))
    mutation(payload["activation"])
    with pytest.raises(targets.ContractError, match="exact pair/strategy/layer proof matrix"):
        targets.finalize_target_manifest(payload)

def test_prepared_support_rejects_positional_rows_and_missing_stable_identity():
    positional = _manifest_payload()
    positional["support"]["splits"]["train"][0] = 0
    with pytest.raises(targets.ContractError, match="keys must be exactly"):
        targets.finalize_target_manifest(positional)

    missing_identity = _manifest_payload()
    del missing_identity["support"]["splits"]["train"][0]["stable_id"]
    with pytest.raises(targets.ContractError, match="keys must be exactly"):
        targets.finalize_target_manifest(missing_identity)


def test_complete_activation_requires_target_bound_refs_for_every_route():
    missing_route = _manifest_payload(layers=3)
    missing_route["activation"]["routes"].pop()
    with pytest.raises(targets.ContractError, match="exact pair/strategy/layer proof matrix"):
        targets.finalize_target_manifest(missing_route)

    cross_target_ref = _manifest_payload(layers=3)
    cross_target_ref["activation"]["routes"][0]["proof_ref"]["uri"] = (
        "cache://proof/other__model/short_bench/chat_first/1/record.json"
    )
    with pytest.raises(
        targets.ContractError,
        match="activation route reference is neither target-scoped nor canonically content-addressed",
    ):
        targets.finalize_target_manifest(cross_target_ref)


def test_blocked_and_finalized_targets_are_excluded_from_new_execution():
    blocked = _manifest_payload(activation_status="partial", blocked=True)
    targets.validate_target_manifest(targets.finalize_target_manifest(blocked))

    invalid_blocked = _manifest_payload(blocked=True)
    invalid_blocked["activation"]["eligible"] = True
    with pytest.raises(targets.ContractError, match="eligibility does not match"):
        targets.finalize_target_manifest(invalid_blocked)

    finalized = targets.finalize_target_manifest(
        _manifest_payload(execution_state="finalized")
    )
    assert finalized["execution"]["rerun_locked"] is True

    unlocked = copy.deepcopy(finalized)
    unlocked["execution"]["rerun_locked"] = False
    unlocked.pop("manifest_sha256")
    with pytest.raises(targets.ContractError, match="rerun lock"):
        targets.finalize_target_manifest(unlocked)

    cross_target = _manifest_payload(execution_state="finalized")
    cross_target["execution"]["publication"]["uri"] = (
        "gs://bucket/results/other__model/short_bench/execution-v3/publication.json"
    )
    with pytest.raises(targets.ContractError, match="publication URI"):
        targets.finalize_target_manifest(cross_target)
