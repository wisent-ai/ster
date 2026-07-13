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

TEST_CODE_REVISION = "b" * 40


class FakeStore:
    """Create-only, generation-addressed object store with a write ledger."""

    def __init__(self, *, fail_at=None):
        self.objects = {}
        self.writes = []
        self.reads = []
        self.create_attempts = 0
        self.fail_at = fail_at
        self.next_generation = 1

    def create(self, uri, data, content_type="application/json"):
        if uri in self.objects:
            raise final.FinalTestError(f"object already exists: {uri}")
        attempt = self.create_attempts
        self.create_attempts += 1
        if self.fail_at == attempt:
            raise final.FinalTestError("injected create failure")
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
        self.reads.append((uri, generation))
        if uri not in self.objects:
            raise final.FinalTestError(f"object missing: {uri}")
        data, observed = self.objects[uri]
        if generation is not None and str(generation) != observed:
            raise final.FinalTestError(f"generation mismatch: {uri}")
        return data, observed


def _runtime():
    return {
        "runtime": "stado-local",
        "python": "3.12.10",
        "torch": "2.7.1",
        "cuda": "12.8",
        "driver": "570.133",
        "gpu": "nvidia-rtx-pro-6000",
        "precision": "bfloat16",
        "evaluator_source_sha256": "e" * 64,
        "coherence_source_sha256": "c" * 64,
        "tokenizer_revision": "a" * 40,
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


_REF_PAYLOADS = {}


def _ref(uri, value, generation="1"):
    payload = final._canonical_bytes(value)
    _REF_PAYLOADS[uri] = payload
    return {"uri": uri, "generation": generation,
            "sha256": hashlib.sha256(payload).hexdigest(), "size": str(len(payload))}


def _calibration():
    methods = {}
    for index, method in enumerate(final.METHODS):
        params = {"layer": index + 1, "strength": 1.0, "extraction_strategy": "chat_first"}
        config_hash = final._canonical_json_sha256(params)
        selected = {"schema_version": 1, "method": method,
                    "best_params": params, "config_sha256": config_hash}
        methods[method] = {
            "params": params, "config_sha256": config_hash,
            "selected_config": _ref(f"gs://cal/{method}/selected", selected),
            "frozen_config": _ref(f"gs://cal/{method}/frozen",
                                  {"schema_version": 2, "method": method, "best_params": params}),
            "provenance": _ref(f"gs://cal/{method}/provenance", {"method": method}),
            "selection_completion": _ref(f"gs://cal/{method}/selection-completion", {"complete": True}),
            "activation_proof": _ref(f"gs://cal/{method}/activation-proof", {
                "complete": True, "extraction_strategy": "chat_first", "layers": [index + 1],
                "pair_ids": list(range(500)),
            }),
            "train_enriched": _ref(f"gs://cal/{method}/train-enriched",
                                   {"num_pairs": 300, "pair_ids": list(range(300))}),
            "_train_pair_ids": list(range(300)),
        }
    index_ref = _ref("gs://cal/calibration-index.json", {"kind": "score-free-calibration-index"}, "9")
    test_payload = {
        "task_name": final.BENCHMARK, "num_pairs": 100, "pair_ids": list(range(400, 500)),
        "pairs": [{"pair_id": pair_id, "stable_id": f"test-{pair_id - 400}",
                   "prompt": f"p{pair_id}", "positive_response": {"model_response": "yes"},
                   "negative_response": {"model_response": "no"}}
                  for pair_id in range(400, 500)],
    }
    return {**index_ref, "model_revision": "a" * 40,
            "test_pairs": _ref("gs://cal/test-pairs", test_payload),
            "test_pair_ids": list(range(400, 500)), "methods": methods}


def _contract():
    return final._build_contract(
        _inventory(), _calibration(), TEST_CODE_REVISION, _runtime(),
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


def _seed_sealed_inputs(store, contract):
    refs = [contract["calibration"]["index"], contract["calibration"]["test_pairs"]]
    for record in contract["calibration"]["methods"].values():
        refs.extend(value for value in record.values()
                    if isinstance(value, dict) and set(value) == {"uri", "generation", "sha256", "size"})
    for ref in refs:
        store.objects[ref["uri"]] = (_REF_PAYLOADS[ref["uri"]], ref["generation"])


def _seal(store, tmp_path, contract=None):
    contract, manifests = _write_bundle(tmp_path, contract)
    _seed_sealed_inputs(store, contract)
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
        "evaluator_version": contract["revisions"]["runtime"]["evaluator_source_sha256"],
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


def _replace_arm_evaluation(store, contract, arm, correctness):
    prefix = f"{contract['publication']['remote_prefix']}runs/{arm}/{contract['contract_sha256']}/"
    completion_uri = prefix + "completion.json"
    completion_bytes, completion_generation = store.objects[completion_uri]
    completion = json.loads(completion_bytes)
    support = contract["split_contract"]["evaluation"]["support"]
    predictions = []
    evaluations = []
    for index, (identity, correct) in enumerate(zip(support, correctness)):
        outcome = {"correct": correct, "confidence": 0.8, "expected_answer": "yes"}
        predictions.append({**identity, "correct": correct, "confidence": 0.8,
                            "evaluation": outcome})
        evaluations.append({"prompt": f"deliberately-non-unique-{index % 2}",
                            "positive_reference": "yes", "negative_reference": "no",
                            "evaluation": outcome})
    correct_count = sum(correctness)
    replacements = {
        "test_predictions.jsonl": b"".join(final._canonical_bytes(row) + b"\n"
                                             for row in predictions),
        "scores.json": final._canonical_bytes({
            "evaluator_used": "log_likelihoods", "num_total": 100,
            "num_evaluated": 100, "num_model_required": 0,
            "aggregated_metrics": {"acc": correct_count / 100},
            "evaluations": evaluations,
        }),
    }
    result_ref = completion["artifacts"]["result.json"]
    result = json.loads(store.objects[result_ref["uri"]][0])
    result.update({"primary_metric": correct_count / 100,
                   "raw_accuracy": correct_count / 100,
                   "correct_count": correct_count})
    replacements["result.json"] = final._canonical_bytes(result)
    for name, payload in replacements.items():
        ref = completion["artifacts"][name]
        store.objects[ref["uri"]] = (payload, store.objects[ref["uri"]][1])
        completion["artifacts"][name] = {
            "uri": ref["uri"], "generation": ref["generation"],
            "sha256": hashlib.sha256(payload).hexdigest(), "size": str(len(payload)),
        }
    completion["completion_sha256"] = final._canonical_json_sha256(
        {key: value for key, value in completion.items() if key != "completion_sha256"}
    )
    store.objects[completion_uri] = (final._canonical_bytes(completion), completion_generation)
    return predictions


def _finalized_flip_publication(store, tmp_path):
    calibration = _calibration()
    test_pairs = json.loads(_REF_PAYLOADS[calibration["test_pairs"]["uri"]])
    test_pairs["pairs"][0]["prompt"] = "same prompt"
    test_pairs["pairs"][1]["prompt"] = "same prompt"
    calibration["test_pairs"] = _ref(calibration["test_pairs"]["uri"], test_pairs)
    contract = final._build_contract(
        _inventory(), calibration, TEST_CODE_REVISION, _runtime(),
        "gs://stado/results/target/final-test-v1/",
    )
    contract, manifests, seal, seal_generation = _seal(store, tmp_path, contract)
    _complete_all(store, contract, manifests, seal)
    baseline = [False, True] + [False] * 98
    predictions = {"baseline": _replace_arm_evaluation(store, contract, "baseline", baseline)}
    method_correctness = {
        "caa": [True, False] + [False] * 98,
        "grom": [False, True, True] + [False] * 97,
    }
    for method in final.METHODS:
        correctness = method_correctness.get(method, baseline)
        predictions[method] = _replace_arm_evaluation(store, contract, method, correctness)
    publication = final._finalize(Namespace(
        seal=f"{contract['publication']['remote_prefix']}control/seal.json",
        seal_generation=seal_generation,
    ), store)
    return contract, publication["publication"], predictions


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


def test_contract_accepts_supplied_commit_without_hardcoded_revision():
    revision = "b" * 40
    contract = final._build_contract(_inventory(), _calibration(), revision, _runtime(),
                                     "gs://stado/results/target/final-test-v1/")
    assert contract["revisions"]["code"] == revision
    with pytest.raises(final.FinalTestError, match="40|revision|commit"):
        final._build_contract(_inventory(), _calibration(), "not-a-commit", _runtime(),
                             "gs://stado/results/target/final-test-v1/")


def test_manifests_seal_route_specific_immutable_inputs_and_distinct_proofs():
    manifests = final._build_manifests(_contract())
    assert all(manifest["test_pairs"]["generation"] for manifest in manifests.values())
    for method in final.METHODS:
        calibration = manifests[method]["calibration"]
        assert calibration["activation_proof"]["uri"] != calibration["selection_completion"]["uri"]
        assert calibration["train_enriched"]["generation"]
        assert calibration["train_enriched"]["size"]


def test_score_free_projection_matches_real_best_and_frozen_outputs(tmp_path):
    methods = {}
    for index, method in enumerate(final.METHODS):
        params = {"layer": index + 1, "strength": 1.0, "extraction_strategy": "chat_first"}
        config_hash = final._canonical_json_sha256(params)
        files = {
            "selected_config": {"schema_version": 1, "method": method,
                                "best_params": params, "config_sha256": config_hash},
            "frozen_config": {"schema_version": 2, "method": method, "best_params": params},
            "provenance": {"method": method},
            "selection_completion": {"complete": True, "selected_config_sha256": config_hash},
            "activation_proof": {"complete": True, "extraction_strategy": "chat_first",
                                 "layers": [index + 1], "pair_ids": list(range(500))},
            "train_enriched": {"num_pairs": 300, "pair_ids": list(range(300))},
        }
        record = {"config_sha256": config_hash}
        for name, value in files.items():
            artifact_path = tmp_path / f"{method}-{name}.json"
            artifact_path.write_bytes(final._canonical_bytes(value))
            record[name] = {"path": str(artifact_path), "uri": f"gs://cal/{method}/{name}",
                            "generation": str(index + 1),
                            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest()}
        methods[method] = record
    test_payload = {
        "task_name": final.BENCHMARK, "num_pairs": 100, "pair_ids": list(range(400, 500)),
        "pairs": [{"pair_id": pair_id, "stable_id": f"test-{pair_id - 400}",
                   "prompt": f"p{pair_id}", "positive_response": {"model_response": "yes"},
                   "negative_response": {"model_response": "no"}}
                  for pair_id in range(400, 500)],
    }
    test_path = tmp_path / "test-pairs.json"
    test_path.write_bytes(final._canonical_bytes(test_payload))
    test_pairs = {"path": str(test_path), "uri": "gs://cal/test-pairs", "generation": "20",
                  "sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
    index = {
        "schema_version": 1,
        "protocol": {"id": final.CALIBRATION_PROTOCOL_ID, "revision": 1,
                     "prior_definitions_sha256": final.PRIOR_DEFINITIONS_SHA256},
        "target": {"model": final.MODEL, "benchmark": final.BENCHMARK, "target_id": final.TARGET_ID},
        "revisions": {"model": "a" * 40, "activation": final.ACTIVATION_REVISION},
        "input_identity": {"pair_text_sha256": final.PAIR_TEXT_SHA256,
                           "full_support_sha256": final.FULL_SUPPORT_SHA256},
        "extraction_strategies": list(final.FORMATS), "trials_per_method": 14,
        "test_evaluations": 0, "test_pairs": test_pairs, "methods": methods,
    }
    path = tmp_path / "index.json"
    path.write_bytes(final._canonical_bytes(index))
    loaded = final._load_calibration_index(path, "9", "gs://cal/index")
    assert loaded["methods"]["caa"]["params"] == {
        "layer": 1, "strength": 1.0, "extraction_strategy": "chat_first"
    }


def test_stado_jobs_pin_manifest_and_seal_generations():
    contract = _contract()
    manifests = final._build_manifests(contract)
    sealed = {arm: _ref(f"gs://sealed/{arm}", manifest, str(index + 1))
              for index, (arm, manifest) in enumerate(manifests.items())}
    seal_ref = _ref("gs://sealed/seal", {"sealed": True}, "99")
    for job in final._build_stado_jobs(contract, sealed, seal_ref):
        command = job["command"]
        assert command[command.index("--manifest-generation") + 1] == sealed[job["arm"]]["generation"]
        assert command[command.index("--seal-generation") + 1] == "99"




def test_calibration_loader_rejects_score_and_validation_leakage_before_contract(tmp_path):
    selected = {"schema_version": 1, "method": "caa", "params": {"layer": 1}, "config_sha256": "x"}
    selected["validation_summary"] = {"best_validation_score": 0.9}
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected))
    digest = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    artifact = {"path": str(selected_path), "uri": "gs://cal/object", "sha256": digest, "generation": "1"}
    methods = {method: {"selected_config": artifact, "frozen_config": artifact,
                        "provenance": artifact, "selection_completion": artifact,
                        "activation_proof": artifact, "train_enriched": artifact,
                        "config_sha256": "0" * 64} for method in final.METHODS}
    test_payload = {"task_name": final.BENCHMARK, "num_pairs": 100,
                    "pair_ids": list(range(400, 500)), "pairs": [{} for _ in range(100)]}
    test_path = tmp_path / "test-pairs.json"
    test_path.write_bytes(final._canonical_bytes(test_payload))
    test_pairs = {"path": str(test_path), "uri": "gs://cal/test-pairs", "generation": "2",
                  "sha256": hashlib.sha256(test_path.read_bytes()).hexdigest()}
    index = {
        "schema_version": 1,
        "protocol": {"id": final.CALIBRATION_PROTOCOL_ID, "revision": 1,
                     "prior_definitions_sha256": final.PRIOR_DEFINITIONS_SHA256},
        "target": {"model": final.MODEL, "benchmark": final.BENCHMARK, "target_id": final.TARGET_ID},
        "revisions": {"model": "a" * 40, "activation": final.ACTIVATION_REVISION},
        "input_identity": {"pair_text_sha256": final.PAIR_TEXT_SHA256,
                           "full_support_sha256": final.FULL_SUPPORT_SHA256},
        "extraction_strategies": list(final.FORMATS), "trials_per_method": 14,
        "test_evaluations": 0, "test_pairs": test_pairs, "methods": methods,
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


def test_seal_generation_reads_every_claimed_input_and_rejects_hash_drift(tmp_path):
    store = FakeStore()
    contract, _ = _write_bundle(tmp_path)
    _seed_sealed_inputs(store, contract)
    expected_refs = [contract["calibration"]["index"], contract["calibration"]["test_pairs"]]
    for record in contract["calibration"]["methods"].values():
        expected_refs.extend(record[name] for name in (
            "selected_config", "frozen_config", "provenance", "selection_completion",
            "activation_proof", "train_enriched"))
    drifted = expected_refs[-1]
    original = store.objects[drifted["uri"]]
    store.objects[drifted["uri"]] = (original[0] + b" ", original[1])
    with pytest.raises(final.FinalTestError, match="bytes differ"):
        final._seal(Namespace(bundle=tmp_path, output=None), store)
    assert {(ref["uri"], ref["generation"]) for ref in expected_refs}.issubset(set(store.reads))
    assert not any(uri.endswith("control/seal.json") for uri in store.writes)


def test_mid_seal_resumes_only_exact_byte_identical_preobjects(tmp_path):
    store = FakeStore(fail_at=3)
    contract, _ = _write_bundle(tmp_path)
    _seed_sealed_inputs(store, contract)
    with pytest.raises(final.FinalTestError, match="injected"):
        final._seal(Namespace(bundle=tmp_path, output=None), store)
    preobjects = list(store.writes)
    assert preobjects and not any(uri.endswith("control/seal.json") for uri in preobjects)
    store.fail_at = None
    output = final._seal(Namespace(bundle=tmp_path, output=None), store)
    assert output["seal"]["uri"].endswith("control/seal.json")
    assert all(store.writes.count(uri) == 1 for uri in preobjects)

    altered = FakeStore(fail_at=3)
    other_dir = tmp_path / "altered"
    other_dir.mkdir()
    other_contract, _ = _write_bundle(other_dir)
    _seed_sealed_inputs(altered, other_contract)
    with pytest.raises(final.FinalTestError, match="injected"):
        final._seal(Namespace(bundle=other_dir, output=None), altered)
    altered.fail_at = None
    uri = altered.writes[0]
    payload, generation = altered.objects[uri]
    altered.objects[uri] = (payload + b" ", generation)
    with pytest.raises(final.FinalTestError, match="different bytes"):
        final._seal(Namespace(bundle=other_dir, output=None), altered)


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


def test_mid_finalize_resumes_only_exact_content_addressed_aggregates(tmp_path):
    store = FakeStore()
    contract, manifests, seal, seal_generation = _seal(store, tmp_path)
    _complete_all(store, contract, manifests, seal)
    args = Namespace(seal=f"{contract['publication']['remote_prefix']}control/seal.json",
                     seal_generation=seal_generation)
    store.fail_at = store.create_attempts + 2
    with pytest.raises(final.FinalTestError, match="injected"):
        final._finalize(args, store)
    aggregate_writes = [uri for uri in store.writes if "/aggregate/" in uri]
    assert len(aggregate_writes) == 2
    assert not any(uri.endswith("publication.json") for uri in aggregate_writes)
    store.fail_at = None
    output = final._finalize(args, store)
    assert output["publication"]["uri"].endswith("publication.json")
    assert all(store.writes.count(uri) == 1 for uri in aggregate_writes)

    altered = FakeStore()
    other_dir = tmp_path / "altered-finalize"
    other_dir.mkdir()
    other_contract, other_manifests, other_seal, other_generation = _seal(altered, other_dir)
    _complete_all(altered, other_contract, other_manifests, other_seal)
    other_args = Namespace(seal=f"{other_contract['publication']['remote_prefix']}control/seal.json",
                           seal_generation=other_generation)
    altered.fail_at = altered.create_attempts + 2
    with pytest.raises(final.FinalTestError, match="injected"):
        final._finalize(other_args, altered)
    altered.fail_at = None
    aggregate_uri = [uri for uri in altered.writes if "/aggregate/" in uri][0]
    payload, generation = altered.objects[aggregate_uri]
    altered.objects[aggregate_uri] = (payload + b" ", generation)
    with pytest.raises(final.FinalTestError, match="different bytes"):
        final._finalize(other_args, altered)


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


def test_diff_reports_identity_joined_flips_and_filters_methods(tmp_path):
    store = FakeStore()
    contract, publication, predictions = _finalized_flip_publication(store, tmp_path)
    args = Namespace(publication=publication["uri"],
                     publication_generation=publication["generation"],
                     method="all", output=None)

    report = final._diff(args, store)

    assert report["schema_version"] == 1
    assert report["protocol_id"] == final.PROTOCOL_ID
    assert report["contract_sha256"] == contract["contract_sha256"]
    assert report["source_publication"] == publication
    assert report["baseline"] == "baseline"
    assert set(report["methods"]) == set(final.METHODS)
    assert report["methods"]["caa"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "unchanged": 98,
        "net_improvement": 0,
        "improved": [{
            "pair_id": 400,
            "stable_id": "test-0",
            "flip": "wrong_to_correct",
            "prompt": "same prompt",
            "expected_answer": "yes",
            "alternative_answer": "no",
            "baseline_prediction": predictions["baseline"][0],
            "method_prediction": predictions["caa"][0],
        }],
        "regressed": [{
            "pair_id": 401,
            "stable_id": "test-1",
            "flip": "correct_to_wrong",
            "prompt": "same prompt",
            "expected_answer": "yes",
            "alternative_answer": "no",
            "baseline_prediction": predictions["baseline"][1],
            "method_prediction": predictions["caa"][1],
        }],
    }
    assert report["methods"]["grom"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 0,
        "unchanged": 99,
        "net_improvement": 1,
        "improved": [{
            "pair_id": 402,
            "stable_id": "test-2",
            "flip": "wrong_to_correct",
            "prompt": "p402",
            "expected_answer": "yes",
            "alternative_answer": "no",
            "baseline_prediction": predictions["baseline"][2],
            "method_prediction": predictions["grom"][2],
        }],
        "regressed": [],
    }
    unchanged = {"wrong_to_correct": 0, "correct_to_wrong": 0, "unchanged": 100,
                 "net_improvement": 0, "improved": [], "regressed": []}
    for method in set(final.METHODS) - {"caa", "grom"}:
        assert report["methods"][method] == unchanged

    single = final._diff(Namespace(
        publication=publication["uri"],
        publication_generation=publication["generation"],
        method="caa", output=tmp_path / "caa-diff.json",
    ), store)
    assert single["methods"] == {"caa": report["methods"]["caa"]}
    assert (tmp_path / "caa-diff.json").read_bytes() == final._canonical_bytes(single)


def test_diff_refuses_completion_ref_drift_without_writing_remote_objects(tmp_path):
    store = FakeStore()
    contract, publication, _ = _finalized_flip_publication(store, tmp_path)
    completion_uri = (contract["publication"]["remote_prefix"] +
                      f"runs/caa/{contract['contract_sha256']}/completion.json")
    payload, generation = store.objects[completion_uri]
    store.objects[completion_uri] = (payload + b" ", generation)
    writes_before_diff = list(store.writes)

    with pytest.raises(final.FinalTestError, match="completion|immutable|hash|bytes"):
        final._diff(Namespace(
            publication=publication["uri"],
            publication_generation=publication["generation"],
            method="caa", output=None,
        ), store)

    assert store.writes == writes_before_diff
