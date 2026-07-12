import copy
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "steering" / "desired_results_final_test.py"
SPEC = importlib.util.spec_from_file_location("desired_results_final_test", PATH)
final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final)


class FakeStore:
    """Create-only, generation-addressed object store with a write ledger."""

    def __init__(self):
        self.objects = {}
        self.writes = []
        self.next_generation = 1

    def create(self, uri, data, content_type="application/json"):
        if uri in self.objects:
            raise final.FinalTestError(f"object already exists: {uri}")
        generation = str(self.next_generation)
        self.next_generation += 1
        payload = bytes(data)
        self.objects[uri] = (payload, generation)
        self.writes.append(uri)
        return {
            "uri": uri,
            "generation": generation,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": str(len(payload)),
        }

    def exists(self, uri):
        return uri in self.objects

    def read(self, uri, generation=None):
        if uri not in self.objects:
            raise final.FinalTestError(f"object missing: {uri}")
        data, observed = self.objects[uri]
        if generation is not None and str(generation) != observed:
            raise final.FinalTestError(f"generation mismatch: {uri}")
        return data, observed


def _runtime():
    return {
        "container": "sha256:" + "d" * 64,
        "python": "3.12.10",
        "torch": "2.7.1",
        "cuda": "12.8",
        "driver": "570.133",
        "gpu": "nvidia-rtx-pro-6000",
        "precision": "bfloat16",
        "evaluator_version": "wisent-log-likelihood-v1",
        "tokenizer_revision": "a" * 40,
        "coherence": {"probe": "winogrande-coherence-v1", "aggregation": "mean"},
    }


def _inventory():
    train = [{"pair_id": i, "stable_id": f"train-{i}"} for i in range(300)]
    test = [{"pair_id": 400 + i, "stable_id": f"test-{i}"} for i in range(100)]
    return {
        "target": {
            "model": final.MODEL,
            "model_slug": final.MODEL_SLUG,
            "benchmark": final.BENCHMARK,
            "target_id": final.TARGET_ID,
            "optimization_run_id": "primary",
        },
        "identity": {
            "pair_text_sha256": final.PAIR_TEXT_SHA256,
            "full_support_sha256": final.FULL_SUPPORT_SHA256,
            "split_assignment_sha256": "1" * 64,
            "train_support_sha256": final._canonical_json_sha256(train),
            "test_support_sha256": final._canonical_json_sha256(test),
        },
        "train": train,
        "test": test,
    }


def _calibration():
    methods = {}
    for index, method in enumerate(final.METHODS):
        params = {"layer": index + 1, "strength": 1.0, "extraction_strategy": "chat_first"}
        methods[method] = {
            "params": params,
            "config_sha256": final._canonical_json_sha256(params),
            "selected_config": {"uri": f"gs://cal/{method}/selected", "generation": "1", "sha256": "1" * 64},
            "frozen_config": {"uri": f"gs://cal/{method}/frozen", "generation": "2", "sha256": "2" * 64},
            "provenance": {"uri": f"gs://cal/{method}/provenance", "generation": "3", "sha256": "3" * 64},
            "completion": {"uri": f"gs://cal/{method}/completion", "generation": "4", "sha256": "4" * 64},
        }
    return {
        "uri": "gs://cal/calibration-index.json",
        "sha256": "5" * 64,
        "generation": "9",
        "model_revision": "a" * 40,
        "methods": methods,
    }


def _contract():
    return final._build_contract(
        _inventory(), _calibration(), final.CODE_REVISION, _runtime(),
        "gs://stado/results/target/final-test-v1/",
    )


def _write_bundle(path, contract=None):
    contract = contract or _contract()
    manifests = final._build_manifests(contract)
    (path / "manifests").mkdir(parents=True)
    (path / "contract.json").write_bytes(final._canonical_bytes(contract))
    for arm, manifest in manifests.items():
        (path / "manifests" / f"{arm}.json").write_bytes(final._canonical_bytes(manifest))
    bundle = {
        "schema_version": 1,
        "contract_sha256": contract["contract_sha256"],
        "contract_file": "contract.json",
        "manifests": {arm: f"manifests/{arm}.json" for arm in final.ARMS},
    }
    (path / "bundle.json").write_bytes(final._canonical_bytes(bundle))
    return contract, manifests


def _seal(store, tmp_path):
    contract, manifests = _write_bundle(tmp_path)
    output = final._seal(Namespace(bundle=tmp_path, output=None), store)
    seal_bytes, seal_generation = store.read(output["seal"]["uri"])
    return contract, manifests, json.loads(seal_bytes), seal_generation


def _artifact(store, uri, value, *, jsonl=False):
    if jsonl:
        data = b"".join(final._canonical_bytes(row) + b"\n" for row in value)
    else:
        data = final._canonical_bytes(value)
    return store.create(uri, data)


def _complete_all(store, contract, manifests, seal, *, mutations=None, count=9):
    shared = {
        "metric_contract_sha256": contract["metric_contract"]["metric_contract_sha256"],
        "test_support_sha256": contract["input_identity"]["test_support_sha256"],
        "ordered_test_ids_sha256": final._canonical_json_sha256(
            [row["pair_id"] for row in contract["split_contract"]["evaluation"]["support"]]
        ),
        "pair_text_sha256": final.PAIR_TEXT_SHA256,
        "model_revision": contract["revisions"]["model"],
        "tokenizer_revision": contract["revisions"]["runtime"]["tokenizer_revision"],
        "code_revision": contract["revisions"]["code"],
        "runtime_identity_sha256": final._canonical_json_sha256(contract["revisions"]["runtime"]),
        "evaluator": "log_likelihoods",
        "evaluator_version": contract["revisions"]["runtime"]["evaluator_version"],
        "evaluation_mode": "log_likelihood",
        "sample_count": 100,
        "aggregation": "arithmetic mean over exact ordered test support",
    }
    for index, arm in enumerate(final.ARMS[:count]):
        result = {
            "schema_version": 1, "arm": arm,
            "primary_metric": 0.50 + index / 100,
            "raw_accuracy": 0.50, "correct_count": 50,
            "num_total": 100, "num_evaluated": 100,
            "num_errors": 0, "num_skipped": 0, **shared,
        }
        if mutations and arm in mutations:
            result.update(mutations[arm])
        prefix = f"{contract['publication']['remote_prefix']}runs/{arm}/{contract['contract_sha256']}/"
        evaluations = []
        predictions = []
        responses = []
        for i, support in enumerate(contract["split_contract"]["evaluation"]["support"]):
            outcome = {"correct": i % 2 == 0, "confidence": 0.8}
            evaluations.append({"prompt": f"p{i}", "positive_reference": "yes",
                                "negative_reference": "no", "evaluation": outcome})
            responses.append({"prompt": f"p{i}", "positive_reference": "yes",
                              "negative_reference": "no"})
            predictions.append({**support, "correct": outcome["correct"],
                                "confidence": outcome["confidence"], "evaluation": outcome})
        scores = {"evaluator_used": "log_likelihoods", "num_total": 100,
                  "num_evaluated": 100, "num_model_required": 0,
                  "aggregated_metrics": {"acc": result["primary_metric"]},
                  "evaluations": evaluations}
        refs = {
            "result.json": _artifact(store, prefix + "result.json", result),
            "test_predictions.jsonl": _artifact(store, prefix + "test_predictions.jsonl", predictions, jsonl=True),
            "scores.json": _artifact(store, prefix + "scores.json", scores),
            "responses.json": _artifact(store, prefix + "responses.json", {"responses": responses}),
            "provenance.json": _artifact(store, prefix + "provenance.json", {"arm": arm}),
        }
        manifest_ref = seal["manifests"][arm]
        completion = {
            "schema_version": 1, "arm": arm,
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifests[arm]["manifest_sha256"],
            "manifest_generation": manifest_ref["generation"],
            "metric_contract_sha256": contract["metric_contract"]["metric_contract_sha256"],
            "artifacts": refs,
        }
        completion["completion_sha256"] = final._canonical_json_sha256(completion)
        store.create(prefix + "completion.json", final._canonical_bytes(completion))


def _replace_predictions(store, contract, seal, arm, predictions):
    prefix = f"{contract['publication']['remote_prefix']}runs/{arm}/{contract['contract_sha256']}/"
    completion_uri = prefix + "completion.json"
    completion_bytes, generation = store.objects[completion_uri]
    completion = json.loads(completion_bytes)
    prediction_uri = prefix + "test_predictions.jsonl"
    payload = b"".join(final._canonical_bytes(row) + b"\n" for row in predictions)
    store.objects[prediction_uri] = (payload, store.objects[prediction_uri][1])
    completion["artifacts"]["test_predictions.jsonl"] = {
        "uri": prediction_uri,
        "generation": store.objects[prediction_uri][1],
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": str(len(payload)),
    }
    completion["completion_sha256"] = final._canonical_json_sha256(
        {key: value for key, value in completion.items() if key != "completion_sha256"}
    )
    store.objects[completion_uri] = (final._canonical_bytes(completion), generation)


def test_contract_uses_distinct_ordered_id_and_full_support_hashes():
    contract = _contract()
    support = contract["split_contract"]["evaluation"]["support"]
    assert contract["input_identity"]["test_support_sha256"] == final._canonical_json_sha256(support)
    expected_ordered = final._canonical_json_sha256([row["pair_id"] for row in support])
    assert expected_ordered != contract["input_identity"]["test_support_sha256"]


def test_contract_seals_exactly_nine_score_free_manifests_and_jobs(tmp_path):
    store = FakeStore()
    contract, manifests, seal, _ = _seal(store, tmp_path)

    assert tuple(manifests) == final.ARMS
    assert tuple(seal["manifests"]) == final.ARMS
    assert contract["split_contract"]["fit"]["count"] == 300
    assert contract["split_contract"]["selection"] == {"name": None, "pair_ids": [], "reads": 0}
    assert contract["split_contract"]["evaluation"]["count"] == 100
    serialized = json.dumps(manifests, sort_keys=True)
    assert "validation-" not in serialized.lower()
    assert "best_validation_score" not in serialized.lower()
    assert "validation_summary" not in serialized.lower()
    assert '"score"' not in serialized.lower()
    assert manifests["baseline"]["calibration"] is None
    assert all(manifests[method]["method"] == method for method in final.METHODS)

    seal_ref = {"uri": "gs://seal", "generation": "1", "sha256": "0" * 64, "size": "1"}
    jobs = final._build_stado_jobs(contract, seal["manifests"], seal_ref)
    assert len(jobs) == 9
    assert {job["arm"] for job in jobs} == set(final.ARMS)
    assert len({job["name"] for job in jobs}) == 9
    assert all(job["maxAttempts"] == 1 and job["retry"] == "forbidden" for job in jobs)
    forbidden = {"--hpo", "--seed", "--trial", "--retry", "--optimizer"}
    assert all(forbidden.isdisjoint(job["command"]) for job in jobs)


def test_calibration_loader_rejects_score_and_validation_leakage_before_contract(tmp_path):
    selected = {"schema_version": 1, "method": "caa", "params": {"layer": 1}, "config_sha256": "x"}
    selected["validation_summary"] = {"best_validation_score": 0.9}
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected))
    digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    artifact = {"path": str(selected_path), "uri": "gs://cal/object", "sha256": digest, "generation": "1"}
    methods = {method: {"selected_config": artifact, "frozen_config": artifact, "provenance": artifact,
                        "completion": artifact, "config_sha256": "0" * 64} for method in final.METHODS}
    index = {
        "schema_version": 1,
        "protocol": {"id": final.CALIBRATION_PROTOCOL_ID, "revision": 1,
                     "prior_definitions_sha256": final.PRIOR_DEFINITIONS_SHA256},
        "target": {"model": final.MODEL, "benchmark": final.BENCHMARK, "target_id": final.TARGET_ID},
        "revisions": {"model": "a" * 40, "activation": final.ACTIVATION_REVISION},
        "input_identity": {"pair_text_sha256": final.PAIR_TEXT_SHA256,
                           "full_support_sha256": final.FULL_SUPPORT_SHA256},
        "extraction_strategies": list(final.FORMATS), "trials_per_method": 14,
        "test_evaluations": 0, "methods": methods,
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))

    with pytest.raises(final.FinalTestError, match="forbidden score/validation"):
        final._load_calibration_index(path, "1")


@pytest.mark.parametrize("already", ["seal", "claim", "completion", "publication"])
def test_seal_is_create_only_and_existing_control_state_refuses(tmp_path, already):
    store = FakeStore()
    contract, _ = _write_bundle(tmp_path)
    prefix = contract["publication"]["remote_prefix"]
    uri = {
        "seal": prefix + "control/seal.json",
        "claim": prefix + "control/claims/caa.json",
        "completion": prefix + f"runs/caa/{contract['contract_sha256']}/completion.json",
        "publication": prefix + "publication.json",
    }[already]
    store.create(uri, b"{}")

    with pytest.raises(final.FinalTestError, match="already exists"):
        final._seal(Namespace(bundle=tmp_path, output=None), store)
    assert prefix + "control/seal.json" not in store.writes or already == "seal"


def test_finalize_refuses_every_partial_completion_count(tmp_path):
    for count in range(9):
        store = FakeStore()
        arm_dir = tmp_path / str(count)
        arm_dir.mkdir()
        contract, manifests, seal, seal_generation = _seal(store, arm_dir)
        _complete_all(store, contract, manifests, seal, count=count)
        before = list(store.writes)
        with pytest.raises(final.FinalTestError, match="object missing"):
            final._finalize(Namespace(seal=f"{contract['publication']['remote_prefix']}control/seal.json",
                                      seal_generation=seal_generation), store)
        assert store.writes == before
        assert f"{contract['publication']['remote_prefix']}publication.json" not in store.objects


@pytest.mark.parametrize("field,bad", [
    ("test_support_sha256", "f" * 64),
    ("ordered_test_ids_sha256", "e" * 64),
    ("pair_text_sha256", "d" * 64),
    ("model_revision", "c" * 40),
    ("tokenizer_revision", "b" * 39 + "c"),
    ("code_revision", "a" * 40),
    ("runtime_identity_sha256", "9" * 64),
    ("evaluator", "generation"),
    ("evaluator_version", "other"),
    ("evaluation_mode", "generation"),
    ("sample_count", 99),
    ("aggregation", "median"),
])
def test_each_comparability_identity_mutation_refuses(tmp_path, field, bad):
    store = FakeStore()
    contract, manifests, seal, seal_generation = _seal(store, tmp_path)
    _complete_all(store, contract, manifests, seal, mutations={"caa": {field: bad}})
    with pytest.raises(final.FinalTestError):
        final._finalize(Namespace(seal=f"{contract['publication']['remote_prefix']}control/seal.json",
                                  seal_generation=seal_generation), store)
    assert f"{contract['publication']['remote_prefix']}publication.json" not in store.objects


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "wrong_stable"])
def test_finalize_refuses_incomplete_duplicate_or_misidentified_predictions(tmp_path, mutation):
    store = FakeStore()
    contract, manifests, seal, seal_generation = _seal(store, tmp_path)
    _complete_all(store, contract, manifests, seal)
    rows = []
    for i in range(100):
        outcome = {"correct": i % 2 == 0, "confidence": 0.8}
        rows.append({"pair_id": 400 + i, "stable_id": f"test-{i}",
                     "correct": outcome["correct"], "confidence": outcome["confidence"],
                     "evaluation": outcome})
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["stable_id"] = "not-the-sealed-stable-id"
    _replace_predictions(store, contract, seal, "caa", rows)

    with pytest.raises(final.FinalTestError, match="prediction|support|100"):
        final._finalize(Namespace(seal=f"{contract['publication']['remote_prefix']}control/seal.json",
                                  seal_generation=seal_generation), store)
    assert f"{contract['publication']['remote_prefix']}publication.json" not in store.objects


def test_finalize_publishes_deterministic_leaderboard_baseline_deltas_and_pointer_last(tmp_path):
    store = FakeStore()
    contract, manifests, seal, seal_generation = _seal(store, tmp_path)
    _complete_all(store, contract, manifests, seal)
    pointer_uri = f"{contract['publication']['remote_prefix']}publication.json"

    output = final._finalize(Namespace(seal=f"{contract['publication']['remote_prefix']}control/seal.json",
                                       seal_generation=seal_generation), store)

    rows = output["leaderboard"]["rows"]
    assert [row["arm"] for row in rows] == list(reversed(final.ARMS))
    baseline = next(row for row in rows if row["arm"] == "baseline")
    assert baseline["baseline_delta"] == 0.0
    assert all(row["baseline_delta"] == pytest.approx(row["primary_metric"] - baseline["primary_metric"])
               for row in rows)
    assert store.writes[-1] == pointer_uri
    assert all("aggregate/" in uri for uri in store.writes[-6:-1])
    with pytest.raises(final.FinalTestError, match="already exists"):
        final._finalize(Namespace(seal=f"{contract['publication']['remote_prefix']}control/seal.json",
                                  seal_generation=seal_generation), store)
