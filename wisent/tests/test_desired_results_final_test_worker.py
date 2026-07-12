import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
FINAL_PATH = ROOT / "scripts" / "steering" / "desired_results_final_test.py"
WORKER_PATH = ROOT / "scripts" / "steering" / "desired_results_final_test_worker.py"
FINAL_SPEC = importlib.util.spec_from_file_location("desired_results_final_test_for_worker_tests", FINAL_PATH)
final = importlib.util.module_from_spec(FINAL_SPEC)
FINAL_SPEC.loader.exec_module(final)
sys.modules["scripts.steering.desired_results_final_test"] = final
WORKER_SPEC = importlib.util.spec_from_file_location("desired_results_final_test_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker)


class FakeStore:
    def __init__(self, *, fail_at=None):
        self.objects = {}
        self.reads = []
        self.writes = []
        self.fail_at = fail_at
        self.generation = 0

    def exists(self, uri):
        return uri in self.objects

    def create(self, uri, data, content_type="application/json"):
        if uri in self.objects:
            raise worker.WorkerError(f"object already exists: {uri}")
        if self.fail_at == len(self.writes):
            raise worker.WorkerError("injected create failure")
        self.generation += 1
        payload = bytes(data)
        self.objects[uri] = (payload, str(self.generation))
        self.writes.append(uri)
        return {"uri": uri, "generation": str(self.generation),
                "sha256": hashlib.sha256(payload).hexdigest(), "size": str(len(payload))}

    def read(self, uri, generation=None):
        self.reads.append(uri)
        if uri not in self.objects:
            raise worker.WorkerError(f"object missing: {uri}")
        data, observed = self.objects[uri]
        if generation is not None and str(generation) != observed:
            raise worker.WorkerError("generation mismatch")
        return data, observed


def _contract_and_manifests():
    train = [{"pair_id": i, "stable_id": f"train-{i}"} for i in range(300)]
    test = [{"pair_id": i + 400, "stable_id": f"test-{i}"} for i in range(100)]
    inventory = {
        "target": {"model": final.MODEL, "model_slug": final.MODEL_SLUG,
                   "benchmark": final.BENCHMARK, "target_id": final.TARGET_ID,
                   "optimization_run_id": "primary"},
        "identity": {"pair_text_sha256": final.PAIR_TEXT_SHA256,
                     "full_support_sha256": final.FULL_SUPPORT_SHA256,
                     "split_assignment_sha256": "1" * 64,
                     "train_support_sha256": final._canonical_json_sha256(train),
                     "test_support_sha256": final._canonical_json_sha256(test)},
        "train": train, "test": test,
    }
    methods = {}
    strategies = ("chat_first", "chat_last", "chat_mean", "chat_max_norm",
                  "chat_weighted", "mc_balanced", "role_play", "chat_first")
    for index, (method, strategy) in enumerate(zip(final.METHODS, strategies), 1):
        params = {"layer": index, "strength": 1.0, "extraction_strategy": strategy}
        methods[method] = {
            "params": params, "config_sha256": final._canonical_json_sha256(params),
            "selected_config": {"uri": f"gs://cal/{method}/selected", "generation": "1", "sha256": "1" * 64},
            "frozen_config": {"uri": f"gs://cal/{method}/frozen", "generation": "2", "sha256": "2" * 64},
            "provenance": {"uri": f"gs://cal/{method}/provenance", "generation": "3", "sha256": "3" * 64},
            "completion": {"uri": f"gs://cal/{method}/completion", "generation": "4", "sha256": "4" * 64},
        }
    calibration = {"uri": "gs://cal/index", "sha256": "5" * 64, "generation": "9",
                   "model_revision": "a" * 40, "methods": methods}
    runtime = {"container": "sha256:" + "d" * 64, "python": "3.12", "torch": "2.7",
               "cuda": "12.8", "driver": "570", "gpu": "nvidia-rtx-pro-6000",
               "precision": "bfloat16", "evaluator_version": "ll-v1",
               "tokenizer_revision": "a" * 40,
               "coherence": {"probe": "fixed", "aggregation": "mean"}}
    contract = final._build_contract(inventory, calibration, final.CODE_REVISION, runtime,
                                     "gs://stado/results/target/final-test-v1/")
    return contract, final._build_manifests(contract)


def _seal_for(manifests):
    refs = {arm: {"uri": f"gs://manifests/{arm}", "generation": "1",
                  "sha256": "9" * 64, "size": "1"} for arm in final.ARMS}
    seal = {"schema_version": 1, "protocol_id": final.PROTOCOL_ID,
            "contract_sha256": manifests["baseline"]["contract_sha256"],
            "contract": {"uri": "gs://contract", "generation": "1", "sha256": "8" * 64, "size": "1"},
            "manifests": refs, "arms": list(final.ARMS),
            "runtime_identity": manifests["baseline"]["revisions"]["runtime"],
            "metric_contract_sha256": manifests["baseline"]["metric_contract"]["metric_contract_sha256"]}
    seal["seal_sha256"] = final._canonical_json_sha256(seal)
    return seal, refs


def _scores(*, evaluator="log_likelihoods", count=100, malformed=None, acc=0.75):
    evaluations = [{"prompt": f"p{i}", "positive_reference": "yes", "negative_reference": "no",
                    "evaluation": {"correct": i % 2 == 0, "confidence": 0.8}} for i in range(count)]
    if malformed is not None and evaluations:
        evaluations[0] = {"prompt": "p0", "positive_reference": "yes",
                          "negative_reference": "no", **malformed}
    return {"evaluator_used": evaluator, "num_total": count, "num_evaluated": count,
            "num_model_required": 0, "aggregated_metrics": {"acc": acc},
            "evaluations": evaluations}

def _responses(count=100):
    return {"responses": [{"prompt": f"p{i}", "positive_reference": "yes",
                            "negative_reference": "no"} for i in range(count)]}


def test_claim_is_permanent_create_only_and_refuses_all_duplicate_terminal_states():
    _, manifests = _contract_and_manifests()
    manifest = manifests["caa"]
    store = FakeStore()
    ref = worker._claim_once(store, manifest)
    assert store.writes == [manifest["claim_uri"]]
    assert ref["uri"] == manifest["claim_uri"]
    with pytest.raises(worker.WorkerError, match="retry forbidden"):
        worker._claim_once(store, manifest)

    root = manifest["output_prefix"].split("/runs/", 1)[0] + "/"
    terminal_uris = (
        manifest["output_prefix"] + "completion.json",
        root + "publication.json",
        root + f"aggregate/{manifest['contract_sha256']}/publication.json",
    )
    for uri in terminal_uris:
        other = FakeStore()
        other.objects[uri] = (b"{}", "1")
        with pytest.raises(worker.WorkerError, match="retry forbidden"):
            worker._claim_once(other, manifest)


def test_execute_claims_before_model_train_test_evaluation_and_completion(monkeypatch, tmp_path):
    _, manifests = _contract_and_manifests()
    manifest = manifests["caa"]
    store = FakeStore()
    manifest_uri = "gs://control/caa-manifest.json"
    manifest_ref = store.create(manifest_uri, final._canonical_bytes(manifest))
    seal, refs = _seal_for(manifests)
    refs["caa"] = manifest_ref
    seal["manifests"] = refs
    seal["seal_sha256"] = final._canonical_json_sha256(
        {key: value for key, value in seal.items() if key != "seal_sha256"}
    )
    seal_uri = "gs://control/seal.json"
    store.create(seal_uri, final._canonical_bytes(seal))
    events = []

    original_claim = worker._claim_once
    def claim(fake_store, value):
        events.append("claim")
        return original_claim(fake_store, value)
    monkeypatch.setattr(worker, "_claim_once", claim)

    from wisent.core.primitives.models import wisent_model as model_module
    class FakeModel:
        def __init__(self, *args, **kwargs): events.append("model")
        def detach(self): events.append("detach")
    monkeypatch.setattr(model_module, "WisentModel", FakeModel)
    monkeypatch.setattr(worker, "_assert_model_revision", lambda model, expected: {"model": expected, "tokenizer": expected})
    monkeypatch.setattr(worker, "_materialize_train", lambda *args: events.append("train") or "train.json")
    monkeypatch.setattr(worker, "_load_test_pairs", lambda *args: events.append("test") or "test.json")

    def run(*args):
        events.append("evaluate")
        work = tmp_path / "run-output"; work.mkdir(exist_ok=True)
        scores = work / "scores.json"; scores.write_text(json.dumps(_scores()))
        responses = work / "responses.json"; responses.write_text(json.dumps(_responses()))
        return {"scores": scores, "responses": responses}
    monkeypatch.setattr(worker, "_run_steered_arm", run)
    monkeypatch.setattr(worker, "_publish_arm", lambda *args: events.append("completion") or {"uri": "gs://completion"})
    monkeypatch.setattr(worker.gc, "collect", lambda: 0)
    import wisent.core.utils.infra_tools.infra as infra_module
    monkeypatch.setattr(infra_module, "empty_device_cache", lambda: None)

    args = SimpleNamespace(manifest=manifest_uri, seal=seal_uri,
                           remote_prefix="gs://stado/results/target/final-test-v1/",
                           output_root=tmp_path, device="cpu")
    worker._execute(args, store)
    assert events[:6] == ["claim", "model", "train", "test", "evaluate", "completion"]


def test_manifest_mutation_and_extra_field_fail_before_claim_or_data_access():
    _, manifests = _contract_and_manifests()
    seal, refs = _seal_for(manifests)
    manifest = manifests["caa"]
    for mutation in (lambda value: value.update(extra=True),
                     lambda value: value["split_contract"]["selection"].update(reads=1),
                     lambda value: value["metric_contract"].update(expected_count=99)):
        changed = copy.deepcopy(manifest)
        mutation(changed)
        with pytest.raises((worker.WorkerError, final.FinalTestError)):
            worker._validate_arm_manifest(changed, seal, refs["caa"],
                                          {"sha256": hashlib.sha256(final._canonical_bytes(seal)).hexdigest()})


def test_all_eight_methods_materialize_only_frozen_route_and_ordered_train(monkeypatch, tmp_path):
    _, manifests = _contract_and_manifests()
    calls = []
    monkeypatch.setattr(worker, "_validate_frozen_params", lambda method, params: None)
    monkeypatch.setattr(worker, "_download_ref", lambda store, ref, destination: destination)
    from wisent.core.reading.modules.utilities.data import enriched_builder

    def build(model, task, layer, strategy, output_dir, pair_ids, **kwargs):
        calls.append((model, task, layer, strategy, list(pair_ids), kwargs))
        path = Path(output_dir) / "enriched.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"pair_ids": pair_ids, "num_pairs": 300}))
        return str(path)

    monkeypatch.setattr(enriched_builder, "build_enriched_from_hf_strict", build)
    for method in final.METHODS:
        worker._materialize_train(FakeStore(), manifests[method], tmp_path / method)

    assert len(calls) == 8
    for method, call in zip(final.METHODS, calls):
        params = manifests[method]["calibration"]["params"]
        assert call[2:4] == (params["layer"], params["extraction_strategy"])
        assert call[4] == list(range(300))
        assert "validation" not in json.dumps(call).lower()


def test_steered_arm_calls_frozen_pipeline_once_with_strict_train_and_test(monkeypatch, tmp_path):
    _, manifests = _contract_and_manifests()
    manifest = manifests["caa"]
    from wisent.core.utils.cli.commands.optimize_steering.pipeline import pipeline
    calls = []
    monkeypatch.setattr(pipeline, "_build_config", lambda method, params: ("CONFIG", 1.0))

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        work = Path(args[3]); work.mkdir(exist_ok=True)
        (work / "scores.json").write_text("{}")
        (work / "responses.json").write_text("{}")
        return SimpleNamespace(details={})

    monkeypatch.setattr(pipeline, "run_pipeline", run)
    worker._run_steered_arm(manifest, object(), "strict-train.json", "test-only.json", tmp_path, "cpu")
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["enriched_pairs_file"] == "strict-train.json"
    assert kwargs["train_pairs_file"] is None
    assert kwargs["test_pairs_file"] == kwargs["evaluation_pairs_file"] == "test-only.json"
    assert kwargs["limit"] is None


def test_baseline_uses_zero_steering_and_exact_log_likelihood_path(monkeypatch, tmp_path):
    _, manifests = _contract_and_manifests()
    from wisent.core.utils.cli.commands.optimize_steering.pipeline import scores as score_module
    events = []

    class Detached:
        def __enter__(self): events.append("detach-enter")
        def __exit__(self, *exc): events.append("detach-exit")

    model = SimpleNamespace(_steering_plan=None, _hook_group=None, detached=lambda: Detached())
    monkeypatch.setattr(score_module, "task_uses_log_likelihoods", lambda task: True)

    def placeholders(input_file, output_file, limit, task, model_name):
        events.append(("placeholder", input_file, limit, task, model_name))
        Path(output_file).write_text(json.dumps(_responses()))

    def evaluate(args):
        events.append(("evaluate", args.cached_model))
        Path(args.output).write_text(json.dumps(_scores()))

    monkeypatch.setattr(score_module, "write_placeholder_responses", placeholders)
    monkeypatch.setattr(score_module, "execute_evaluate_responses", evaluate)
    monkeypatch.setattr(worker, "_evaluation_args",
                        lambda input_file, output_file, cached: SimpleNamespace(input=input_file, output=output_file,
                                                                               cached_model=cached))
    worker._run_baseline_arm(manifests["baseline"], model, "test.json", tmp_path)
    assert events == [("placeholder", "test.json", 100, final.BENCHMARK, final.MODEL),
                      "detach-enter", ("evaluate", model), "detach-exit"]
    assert worker._steering_empty(model)


def test_pinned_revision_uses_resolved_properties_when_tokenizer_has_only_commit_attribute():
    revision = "a" * 40
    model = SimpleNamespace(requested_revision=revision,
                            resolved_model_revision=revision,
                            resolved_tokenizer_revision=revision,
                            tokenizer=SimpleNamespace(_commit_hash=revision))
    assert worker._assert_model_revision(model, revision) == {
        "model": revision, "tokenizer": revision,
    }


@pytest.mark.parametrize("requested,resolved_model,resolved_tokenizer", [
    ("b" * 40, "a" * 40, "a" * 40),
    ("a" * 40, "b" * 40, "a" * 40),
    ("a" * 40, "a" * 40, None),
])
def test_pinned_revision_enforcement_rejects_any_unresolved_identity(requested, resolved_model, resolved_tokenizer):
    model = SimpleNamespace(requested_revision=requested,
                            resolved_model_revision=resolved_model,
                            resolved_tokenizer_revision=resolved_tokenizer)
    with pytest.raises(worker.WorkerError, match="revision|commit"):
        worker._assert_model_revision(model, "a" * 40)


def test_normalize_requires_exact_100_predictions_and_preserves_order():
    _, manifests = _contract_and_manifests()
    result, predictions = worker._normalize_result(manifests["baseline"], _scores(), _responses())
    assert result["num_total"] == result["num_evaluated"] == result["sample_count"] == 100
    assert result["correct_count"] == 50
    assert [row["pair_id"] for row in predictions] == list(range(400, 500))
    assert len(predictions) == 100


@pytest.mark.parametrize("scores,responses,match", [
    (_scores(evaluator="generation"), _responses(), "non-log-likelihood"),
    (_scores(count=99), _responses(99), "100/100"),
    (_scores(malformed={"evaluation": {"confidence": 0.5}}), _responses(), "boolean correctness"),
    (_scores(malformed={"evaluation": {"correct": True, "confidence": float("nan")}}), _responses(), "finite"),
    (_scores(acc=float("inf")), _responses(), "accuracy"),
    (_scores(), _responses(99), "100 rows"),
])
def test_evaluator_and_malformed_outputs_fail_closed(scores, responses, match):
    _, manifests = _contract_and_manifests()
    with pytest.raises(worker.WorkerError, match=match):
        worker._normalize_result(manifests["baseline"], scores, responses)


def test_completion_is_last_create_and_artifact_failure_never_exposes_it(tmp_path):
    _, manifests = _contract_and_manifests()
    manifest = manifests["baseline"]
    scores = tmp_path / "scores-input.json"; scores.write_text(json.dumps(_scores()))
    responses = tmp_path / "responses-input.json"; responses.write_text(json.dumps(_responses()))
    result, predictions = worker._normalize_result(manifest, _scores(), _responses())
    ref = {"sha256": manifest["manifest_sha256"], "generation": "1"}

    store = FakeStore()
    (tmp_path / "ok").mkdir()
    worker._publish_arm(store, manifest, tmp_path / "ok", result, predictions, scores, responses,
                        {"test_reads": 1}, ref)
    assert store.writes[-1].endswith("completion.json")
    assert all(not uri.endswith("completion.json") for uri in store.writes[:-1])

    failing = FakeStore(fail_at=3)
    (tmp_path / "fail").mkdir()
    with pytest.raises(worker.WorkerError, match="injected"):
        worker._publish_arm(failing, manifest, tmp_path / "fail", result, predictions, scores, responses,
                            {"test_reads": 1}, ref)
    assert not any(uri.endswith("completion.json") for uri in failing.writes)


def test_train_and_test_helpers_never_read_validation_support(monkeypatch, tmp_path):
    _, manifests = _contract_and_manifests()
    manifest = manifests["caa"]
    monkeypatch.setattr(worker, "_validate_frozen_params", lambda method, params: None)
    store = FakeStore()
    completion = manifest["calibration"]["completion"]
    payload = b"proof"
    completion.update(sha256=hashlib.sha256(payload).hexdigest())
    store.objects[completion["uri"]] = (payload, completion["generation"])
    from wisent.core.reading.modules.utilities.data import enriched_builder
    from wisent.core.reading.modules.utilities.data.sources.hf import hf_loaders

    def build(model, task, layer, strategy, output_dir, pair_ids, **kwargs):
        path = Path(output_dir) / "rows.json"; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"pair_ids": pair_ids, "num_pairs": 300}))
        return str(path)

    seen_ids = []
    def load(task, ids):
        seen_ids.extend(ids)
        return {pair_id: {"prompt": "p", "positive": "yes", "negative": "no"} for pair_id in ids}

    monkeypatch.setattr(enriched_builder, "build_enriched_from_hf_strict", build)
    monkeypatch.setattr(hf_loaders, "load_pair_texts_from_hf_strict", load)
    worker._materialize_train(store, manifest, tmp_path / "train")
    worker._load_test_pairs(manifest, tmp_path / "test.json")
    assert store.reads == [completion["uri"]]
    assert seen_ids == list(range(400, 500))
    assert not ({*range(300, 400)} & set(seen_ids))
