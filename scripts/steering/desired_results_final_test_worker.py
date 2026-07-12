#!/usr/bin/env python3
"""Execute one permanently claimed arm of the immutable desired-results final test."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from scripts.steering.desired_results_final_test import (
    ACTIVATION_REVISION, ARMS, BENCHMARK, CALIBRATION_PROTOCOL_ID, CODE_REVISION,
    FinalTestError, FULL_SUPPORT_SHA256, GCSStore, MODEL, PAIR_TEXT_SHA256, PROTOCOL_ID,
    TARGET_ID, _atomic_json, _canonical_bytes, _canonical_json_sha256,
    _require_exact_keys, _strict_json_bytes,
)


class WorkerError(FinalTestError):
    """One final-test arm cannot safely execute."""


def _read_object(store: GCSStore, uri: str, generation: str | None = None) -> tuple[Any, Dict[str, str]]:
    data, observed = store.read(uri, generation)
    return _strict_json_bytes(data, uri), {
        "uri": uri, "generation": observed, "sha256": hashlib.sha256(data).hexdigest(),
        "size": str(len(data)),
    }


def _without_hash(value: Mapping[str, Any], field: str) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _validate_arm_manifest(manifest: Mapping[str, Any], seal: Mapping[str, Any],
                           manifest_ref: Mapping[str, Any] | None = None,
                           seal_ref: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    manifest_keys = {"schema_version", "protocol", "target", "contract_sha256", "revisions",
                     "input_identity", "split_contract", "metric_contract", "test_evaluations",
                     "arm", "method", "calibration", "claim_uri", "output_prefix", "manifest_sha256"}
    _require_exact_keys(manifest, manifest_keys, "arm manifest")
    seal_keys = {"schema_version", "protocol_id", "contract", "contract_sha256",
                 "manifests", "arms", "runtime_identity", "metric_contract_sha256", "seal_sha256"}
    _require_exact_keys(seal, seal_keys, "seal")
    if manifest["schema_version"] != 1 or seal["schema_version"] != 1:
        raise WorkerError("manifest and seal must use schema_version 1")
    manifest_hash = _canonical_json_sha256(_without_hash(manifest, "manifest_sha256"))
    if manifest["manifest_sha256"] != manifest_hash:
        raise WorkerError("manifest canonical identity hash differs")
    if seal["seal_sha256"] != _canonical_json_sha256(_without_hash(seal, "seal_sha256")):
        raise WorkerError("seal canonical identity hash differs")
    if seal["protocol_id"] != PROTOCOL_ID or seal["arms"] != list(ARMS) or set(seal["manifests"]) != set(ARMS):
        raise WorkerError("seal does not cover exactly all nine final-test arms")
    if manifest["contract_sha256"] != seal["contract_sha256"]:
        raise WorkerError("manifest contract is not the contract sealed for all arms")
    arm = manifest["arm"]
    if arm not in ARMS or manifest["method"] != (None if arm == "baseline" else arm):
        raise WorkerError("arm/method identity differs")
    if manifest_ref is not None:
        covered = seal["manifests"][arm]
        for key in ("uri", "generation", "sha256", "size"):
            if covered.get(key) != manifest_ref.get(key):
                raise WorkerError(f"seal manifest reference differs on {key}")
    if seal_ref is not None and seal_ref.get("sha256") != hashlib.sha256(_canonical_bytes(seal)).hexdigest():
        raise WorkerError("seal object bytes differ from supplied identity")
    protocol = manifest["protocol"]
    if (protocol.get("id") != PROTOCOL_ID or protocol.get("revision") != 1 or
            protocol.get("calibration_protocol_id") != CALIBRATION_PROTOCOL_ID):
        raise WorkerError("manifest protocol differs")
    _require_exact_keys(manifest["target"], {"model", "model_slug", "benchmark", "target_id", "optimization_run_id"}, "manifest target")
    if manifest["target"].get("model") != MODEL or manifest["target"].get("benchmark") != BENCHMARK or manifest["target"].get("target_id") != TARGET_ID:
        raise WorkerError("manifest target differs")
    revisions = _require_exact_keys(
        manifest["revisions"], {"code", "model", "activation", "runtime"}, "manifest revisions")
    if (revisions["code"] != CODE_REVISION or revisions["activation"] != ACTIVATION_REVISION or
            seal["runtime_identity"] != revisions["runtime"] or
            revisions["runtime"].get("tokenizer_revision") != revisions["model"]):
        raise WorkerError("manifest revision/runtime identity differs from seal")
    if manifest["test_evaluations"] != 1:
        raise WorkerError("each arm must evaluate test exactly once")
    split = manifest["split_contract"]
    _require_exact_keys(split, {"fit", "selection", "evaluation", "validation_pair_ids_forbidden"}, "split contract")
    fit, selection, evaluation = split["fit"], split["selection"], split["evaluation"]
    if (fit.get("name") != "train" or fit.get("count") != 300 or
            selection != {"name": None, "pair_ids": [], "reads": 0} or
            evaluation.get("name") != "test" or evaluation.get("count") != 100 or
            evaluation.get("evaluations_per_arm") != 1 or split["validation_pair_ids_forbidden"] is not True):
        raise WorkerError("manifest must fit train only and evaluate exact test once")
    for label, rows, count in (("train", fit.get("support"), 300), ("test", evaluation.get("support"), 100)):
        if not isinstance(rows, list) or len(rows) != count:
            raise WorkerError(f"{label} support count differs")
        ids = []
        stable = []
        for row in rows:
            _require_exact_keys(row, {"pair_id", "stable_id"}, f"{label} support row")
            if type(row["pair_id"]) is not int or not isinstance(row["stable_id"], str) or not row["stable_id"]:
                raise WorkerError(f"{label} support identity malformed")
            ids.append(row["pair_id"]); stable.append(row["stable_id"])
        if len(ids) != len(set(ids)) or len(stable) != len(set(stable)):
            raise WorkerError(f"{label} support contains duplicate identity")
    if set(row["pair_id"] for row in fit["support"]) & set(row["pair_id"] for row in evaluation["support"]):
        raise WorkerError("train and test support overlap")
    identity = manifest["input_identity"]
    _require_exact_keys(identity, {"pair_text_sha256", "full_support_sha256", "split_assignment_sha256", "train_support_sha256", "test_support_sha256"}, "input identity")
    if identity["pair_text_sha256"] != PAIR_TEXT_SHA256 or identity["full_support_sha256"] != FULL_SUPPORT_SHA256:
        raise WorkerError("manifest input identity differs from frozen support")
    if (identity.get("train_support_sha256") != _canonical_json_sha256(fit["support"]) or
            identity.get("test_support_sha256") != _canonical_json_sha256(evaluation["support"])):
        raise WorkerError("manifest support hashes differ")
    metric = manifest["metric_contract"]
    metric_hash = metric.get("metric_contract_sha256")
    if (metric_hash != _canonical_json_sha256(_without_hash(metric, "metric_contract_sha256")) or
            seal["metric_contract_sha256"] != metric_hash or metric.get("evaluator") != "log_likelihoods" or
            metric.get("expected_count") != 100):
        raise WorkerError("metric contract differs from sealed log-likelihood contract")
    if arm == "baseline":
        if manifest["calibration"] is not None:
            raise WorkerError("baseline must not have calibration configuration")
    else:
        calibration = manifest["calibration"]
        required_cal = {"params", "config_sha256", "selected_config", "frozen_config", "provenance", "completion"}
        _require_exact_keys(calibration, required_cal, "arm calibration")
        if calibration["config_sha256"] != _canonical_json_sha256(calibration["params"]):
            raise WorkerError("frozen parameter hash differs")
        for name in ("selected_config", "frozen_config", "provenance", "completion"):
            _require_exact_keys(calibration[name], {"uri", "sha256", "generation"}, f"calibration {name}")
    prefix = manifest["output_prefix"]
    expected_tail = f"/runs/{arm}/{manifest['contract_sha256']}/"
    if not isinstance(prefix, str) or not prefix.startswith("gs://") or not prefix.endswith(expected_tail) or ".." in prefix:
        raise WorkerError("unsafe arm output prefix")
    if manifest["claim_uri"] != prefix.split("/runs/", 1)[0] + f"/control/claims/{arm}.json":
        raise WorkerError("claim URI is not target/protocol stable")
    return dict(manifest)


def _claim_once(store: GCSStore, manifest: Mapping[str, Any]) -> Dict[str, str]:
    prefix = manifest["output_prefix"].split("/runs/", 1)[0] + "/"
    terminal_paths = [manifest["claim_uri"], manifest["output_prefix"] + "completion.json",
                      prefix + "publication.json",
                      prefix + f"aggregate/{manifest['contract_sha256']}/publication.json"]
    if any(store.exists(uri) for uri in terminal_paths):
        raise WorkerError("claim/completion/aggregate/publication already exists; retry forbidden")
    claim = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "arm": manifest["arm"],
             "contract_sha256": manifest["contract_sha256"],
             "manifest_sha256": manifest["manifest_sha256"], "attempt": 1,
             "permanent": True, "claim_before": ["model_load", "train_fit", "test_read"]}
    claim["claim_sha256"] = _canonical_json_sha256(claim)
    return store.create(manifest["claim_uri"], _canonical_bytes(claim))


def _download_ref(store: GCSStore, ref: Mapping[str, Any], destination: Path) -> Path:
    data, observed = store.read(ref["uri"], ref["generation"])
    if observed != ref["generation"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise WorkerError(f"immutable calibration object drift: {ref['uri']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return destination


def _validate_frozen_params(method: str, params: Mapping[str, Any]) -> None:
    # Importing the runner is safe here: it does not construct a model or optimizer at import time.
    from scripts.steering.desired_results_runner import (
        _bounded_prior_definitions, _canonical_json_sha256 as runner_hash, _validate_sample,
    )
    definitions = _bounded_prior_definitions()
    if runner_hash(definitions) != manifest_prior_hash():
        raise WorkerError("observed bounded prior definitions drifted")
    try:
        _validate_sample(method, dict(params), definitions[method])
    except (KeyError, ValueError, FinalTestError, RuntimeError) as exc:
        raise WorkerError(f"frozen parameters violate bounded prior: {exc}") from exc


def manifest_prior_hash() -> str:
    from scripts.steering.desired_results_final_test import PRIOR_DEFINITIONS_SHA256
    return PRIOR_DEFINITIONS_SHA256


def _materialize_train(store: GCSStore, manifest: Mapping[str, Any], work_dir: Path) -> str:
    calibration = manifest["calibration"]
    params = calibration["params"]
    method = manifest["method"]
    _validate_frozen_params(method, params)
    strategy = params.get("extraction_strategy")
    layer = params.get("layer", params.get("sensor_layer"))
    if not isinstance(strategy, str) or type(layer) is not int:
        raise WorkerError("frozen activation route lacks exact strategy/layer")
    proof = _download_ref(store, calibration["completion"], work_dir / "activation-completion.json")
    from wisent.core.reading.modules.utilities.data.enriched_builder import build_enriched_from_hf_strict
    train_ids = [row["pair_id"] for row in manifest["split_contract"]["fit"]["support"]]
    output = build_enriched_from_hf_strict(
        MODEL, BENCHMARK, layer, strategy, str(work_dir / "strict-train"), train_ids,
        completion_manifest_file=str(proof),
    )
    data = _strict_json_bytes(Path(output).read_bytes(), "strict train activations")
    if data.get("pair_ids") != train_ids or data.get("num_pairs") != 300:
        raise WorkerError("strict train materialization changed ordered support")
    return output


def _load_test_pairs(manifest: Mapping[str, Any], destination: Path) -> str:
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import load_pair_texts_from_hf_strict
    rows = manifest["split_contract"]["evaluation"]["support"]
    ids = [row["pair_id"] for row in rows]
    texts = load_pair_texts_from_hf_strict(BENCHMARK, ids)
    if list(texts) != ids:
        raise WorkerError("strict test text loader changed ordered support")
    pairs = [{"pair_id": pair_id, "stable_id": row["stable_id"], "prompt": texts[pair_id]["prompt"],
              "positive_response": {"model_response": texts[pair_id]["positive"]},
              "negative_response": {"model_response": texts[pair_id]["negative"]}}
             for pair_id, row in zip(ids, rows)]
    _atomic_json(destination, {"task_name": BENCHMARK, "num_pairs": 100,
                               "pair_ids": ids, "pairs": pairs})
    return str(destination)


def _steering_empty(model: Any) -> bool:
    plan = getattr(model, "_steering_plan", None)
    hooks = getattr(model, "_hook_group", None)
    plan_empty = plan is None or (hasattr(plan, "is_empty") and plan.is_empty())
    if hooks is None:
        hooks_empty = True
    elif hasattr(hooks, "__len__"):
        hooks_empty = len(hooks) == 0
    else:
        hooks_empty = not bool(getattr(hooks, "handles", []))
    return bool(plan_empty and hooks_empty)


def _assert_model_revision(model: Any, expected: str) -> Dict[str, str]:
    if getattr(model, "requested_revision", None) != expected:
        raise WorkerError("WisentModel did not preserve requested immutable revision")
    model_commit = getattr(model, "resolved_model_revision", None)
    tokenizer_commit = getattr(model, "resolved_tokenizer_revision", None)
    if model_commit != expected or tokenizer_commit != expected:
        raise WorkerError("observed model/tokenizer commit does not equal sealed revision")
    return {"model": model_commit, "tokenizer": tokenizer_commit}


def _run_steered_arm(manifest: Mapping[str, Any], model: Any, enriched: str,
                      test_pairs: str, work_dir: Path, device: str | None) -> Dict[str, Any]:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import _build_config, run_pipeline
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.scores import task_uses_log_likelihoods
    if not task_uses_log_likelihoods(BENCHMARK):
        raise WorkerError("Winogrande evaluator is not log_likelihoods")
    if not _steering_empty(model):
        raise WorkerError("steered model has pre-existing steering state")
    params = dict(manifest["calibration"]["params"])
    before = _canonical_json_sha256(params)
    config, strength = _build_config(manifest["method"], params)
    pipeline_dir = work_dir / "pipeline"
    pipeline_dir.mkdir()
    result = run_pipeline(MODEL, BENCHMARK, config, str(pipeline_dir), strength,
                          limit=None, device=device, enriched_pairs_file=enriched,
                          train_pairs_file=None, test_pairs_file=test_pairs,
                          evaluation_pairs_file=test_pairs, cached_model=model)
    if not _steering_empty(model):
        raise WorkerError("steered model retained steering state after evaluation")
    if _canonical_json_sha256(params) != before or before != manifest["calibration"]["config_sha256"]:
        raise WorkerError("frozen configuration mutated during final test")
    return {"details": result.details, "scores": pipeline_dir / "scores.json",
            "responses": pipeline_dir / "responses.json"}


def _evaluation_args(input_file: str, output_file: str, model: Any) -> Any:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import _make_args
    from wisent.core.utils.config_tools.constants import (
        EVAL_F1_THRESHOLD, EVAL_GENERATION_EMBEDDING_WEIGHT, EVAL_GENERATION_NLI_WEIGHT,
        SCORE_MIDPOINT_PCT, SPLIT_RATIO_TRAIN_DEFAULT,
    )
    from wisent.core.utils.infra_tools.infra.core.hardware import subprocess_timeout_s
    return _make_args(input=input_file, output=output_file, task=BENCHMARK, verbose=False,
                      f1_threshold=EVAL_F1_THRESHOLD,
                      generation_embedding_weight=EVAL_GENERATION_EMBEDDING_WEIGHT,
                      generation_nli_weight=EVAL_GENERATION_NLI_WEIGHT,
                      train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
                      subprocess_timeout=subprocess_timeout_s(),
                      personalization_good_threshold=SCORE_MIDPOINT_PCT,
                      cached_model=model)


def _run_baseline_arm(manifest: Mapping[str, Any], model: Any, test_pairs: str,
                      work_dir: Path) -> Dict[str, Any]:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.scores import (
        execute_evaluate_responses, task_uses_log_likelihoods, write_placeholder_responses,
    )
    if not task_uses_log_likelihoods(BENCHMARK):
        raise WorkerError("Winogrande evaluator is not log_likelihoods")
    if not _steering_empty(model):
        raise WorkerError("baseline model has steering state before evaluation")
    responses = work_dir / "responses.json"
    scores = work_dir / "scores.json"
    write_placeholder_responses(test_pairs, str(responses), 100, BENCHMARK, MODEL)
    with model.detached():
        execute_evaluate_responses(_evaluation_args(str(responses), str(scores), model))
    if not _steering_empty(model):
        raise WorkerError("baseline model has steering state after evaluation")
    return {"details": _strict_json_bytes(scores.read_bytes(), "baseline scores"),
            "scores": scores, "responses": responses}


def _normalize_result(manifest: Mapping[str, Any], scores: Mapping[str, Any],
                      responses: Mapping[str, Any]) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    if scores.get("evaluator_used") != "log_likelihoods":
        raise WorkerError("final test used a non-log-likelihood evaluator")
    if scores.get("num_total") != 100 or scores.get("num_evaluated") != 100 or scores.get("num_model_required") != 0:
        raise WorkerError("final test did not evaluate exactly 100/100 rows")
    evaluations = scores.get("evaluations")
    response_rows = responses.get("responses")
    support = manifest["split_contract"]["evaluation"]["support"]
    if not isinstance(evaluations, list) or len(evaluations) != 100 or not isinstance(response_rows, list) or len(response_rows) != 100:
        raise WorkerError("scores/responses do not preserve exactly 100 rows")
    predictions = []
    correct_count = 0
    confidences = []
    for index, (evaluation, response_row) in enumerate(zip(evaluations, response_rows), 1):
        if not isinstance(evaluation, dict) or not isinstance(response_row, dict):
            raise WorkerError(f"malformed evaluation/response row {index}")
        if "error" in evaluation:
            raise WorkerError("evaluation contains a top-level error row")
        for key in ("prompt", "positive_reference", "negative_reference"):
            if evaluation.get(key) != response_row.get(key):
                raise WorkerError("evaluator changed ordered response identity")
    for support_row, evaluation in zip(support, evaluations):
        if not isinstance(evaluation, dict) or not isinstance(evaluation.get("evaluation"), dict):
            raise WorkerError("malformed evaluation row")
        outcome = evaluation["evaluation"]
        if "error" in outcome or type(outcome.get("correct")) is not bool:
            raise WorkerError("evaluation contains an error/skip or lacks boolean correctness")
        confidence = outcome.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise WorkerError("evaluation confidence is not finite")
        correct_count += int(outcome["correct"]); confidences.append(float(confidence))
        predictions.append({"pair_id": support_row["pair_id"], "stable_id": support_row["stable_id"],
                            "correct": outcome["correct"], "confidence": float(confidence),
                            "evaluation": outcome})
    acc = scores.get("aggregated_metrics", {}).get("acc")
    if isinstance(acc, bool) or not isinstance(acc, (int, float)) or not math.isfinite(acc):
        raise WorkerError("coherence-adjusted accuracy is not finite")
    raw = correct_count / 100.0
    coherence = float(acc) / raw if raw else 1.0
    runtime = manifest["revisions"]["runtime"]
    result = {
        "schema_version": 1, "arm": manifest["arm"], "method": manifest["method"],
        "primary_metric": float(acc), "raw_accuracy": raw, "correct_count": correct_count,
        "mean_confidence": sum(confidences) / 100.0, "coherence_factor": coherence,
        "num_total": 100, "num_evaluated": 100, "num_errors": 0, "num_skipped": 0,
        "metric_contract_sha256": manifest["metric_contract"]["metric_contract_sha256"],
        "test_support_sha256": manifest["input_identity"]["test_support_sha256"],
        "ordered_test_ids_sha256": _canonical_json_sha256([row["pair_id"] for row in support]),
        "pair_text_sha256": manifest["input_identity"]["pair_text_sha256"],
        "model_revision": manifest["revisions"]["model"],
        "tokenizer_revision": runtime["tokenizer_revision"], "code_revision": manifest["revisions"]["code"],
        "runtime_identity_sha256": _canonical_json_sha256(runtime), "evaluator": "log_likelihoods",
        "evaluator_version": runtime["evaluator_version"], "evaluation_mode": "log_likelihood",
        "sample_count": 100, "aggregation": manifest["metric_contract"]["aggregation"],
    }
    return result, predictions


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _publish_arm(store: GCSStore, manifest: Mapping[str, Any], work_dir: Path,
                 result: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]],
                 scores_path: Path, responses_path: Path, provenance: Mapping[str, Any],
                 manifest_ref: Mapping[str, Any]) -> Dict[str, str]:
    stage = work_dir / "publish"
    stage.mkdir()
    _atomic_json(stage / "result.json", result)
    _atomic_json(stage / "scores.json", _strict_json_bytes(scores_path.read_bytes(), "scores"))
    _atomic_json(stage / "responses.json", _strict_json_bytes(responses_path.read_bytes(), "responses"))
    _atomic_json(stage / "provenance.json", provenance)
    predictions_path = stage / "test_predictions.jsonl"
    with predictions_path.open("x", encoding="ascii") as stream:
        for row in predictions:
            stream.write(_canonical_bytes(row).decode("ascii") + "\n")
        stream.flush(); os.fsync(stream.fileno())
    names = ("result.json", "test_predictions.jsonl", "scores.json", "responses.json", "provenance.json")
    refs = {}
    for name in names:
        path = stage / name
        _fsync_file(path)
        refs[name] = store.create(manifest["output_prefix"] + name, path.read_bytes(),
                                  "application/x-ndjson" if name.endswith(".jsonl") else "application/json")
    completion = {"schema_version": 1, "arm": manifest["arm"],
                  "contract_sha256": manifest["contract_sha256"],
                  "manifest_sha256": manifest["manifest_sha256"],
                  "manifest_generation": manifest_ref["generation"],
                  "metric_contract_sha256": manifest["metric_contract"]["metric_contract_sha256"],
                  "artifacts": refs}
    completion["completion_sha256"] = _canonical_json_sha256(completion)
    return store.create(manifest["output_prefix"] + "completion.json", _canonical_bytes(completion))


def _execute(args: argparse.Namespace, store: GCSStore | None = None) -> Dict[str, Any]:
    store = store or GCSStore()
    manifest, manifest_ref = _read_object(store, args.manifest)
    seal, seal_ref = _read_object(store, args.seal)
    manifest = _validate_arm_manifest(manifest, seal, manifest_ref, seal_ref)
    expected_prefix = args.remote_prefix.rstrip("/") + "/"
    if manifest["output_prefix"].split("runs/", 1)[0] != expected_prefix:
        raise WorkerError("CLI remote prefix differs from sealed output prefix")
    claim_ref = _claim_once(store, manifest)  # permanent and before model/train/test
    work_dir = Path(tempfile.mkdtemp(prefix=f"final-test-{manifest['arm']}-", dir=args.output_root.resolve()))
    try:
        from wisent.core.primitives.models.wisent_model import WisentModel
        model = WisentModel(MODEL, device=args.device, revision=manifest["revisions"]["model"])
        resolved = _assert_model_revision(model, manifest["revisions"]["model"])
        if manifest["arm"] == "baseline":
            test_pairs = _load_test_pairs(manifest, work_dir / "test-pairs.json")
            run = _run_baseline_arm(manifest, model, test_pairs, work_dir)
            fit_ledger = {"fit": "none", "fit_pair_ids": []}
        else:
            enriched = _materialize_train(store, manifest, work_dir)
            test_pairs = _load_test_pairs(manifest, work_dir / "test-pairs.json")
            run = _run_steered_arm(manifest, model, enriched, test_pairs, work_dir, args.device)
            fit_ledger = {"fit": "train_only", "fit_pair_ids": [row["pair_id"] for row in manifest["split_contract"]["fit"]["support"]]}
        scores = _strict_json_bytes(Path(run["scores"]).read_bytes(), "scores")
        responses = _strict_json_bytes(Path(run["responses"]).read_bytes(), "responses")
        result, predictions = _normalize_result(manifest, scores, responses)
        provenance = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
                      "contract_sha256": manifest["contract_sha256"],
                      "manifest_sha256": manifest["manifest_sha256"], "claim": claim_ref,
                      "revisions": manifest["revisions"], "resolved_revisions": resolved,
                      "calibration": manifest["calibration"], **fit_ledger,
                      "test_pair_ids_read": [row["pair_id"] for row in manifest["split_contract"]["evaluation"]["support"]],
                      "test_reads": 1, "optimization_calls": 0, "configuration_mutation": False,
                      "runtime_observed": {"python": platform.python_version()}}
        completion = _publish_arm(store, manifest, work_dir, result, predictions,
                                  Path(run["scores"]), Path(run["responses"]), provenance, manifest_ref)
        return {"arm": manifest["arm"], "claim": claim_ref, "completion": completion}
    finally:
        try:
            if "model" in locals():
                model.detach()
                del model
            gc.collect()
            from wisent.core.utils.infra_tools.infra import empty_device_cache
            empty_device_cache()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        print(json.dumps(_execute(args), sort_keys=True, allow_nan=False))
        return 0
    except (WorkerError, FinalTestError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"desired-results final-test worker refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
