#!/usr/bin/env python3
"""Seal and publish the immutable desired-results final test.

This is control-plane code: it deliberately has no Wisent/model imports.  ``prepare``
validates local immutable inputs and writes a content-addressed bundle; ``seal``
performs create-only GCS writes and emits the nine Stado command specifications;
``finalize`` publishes a leaderboard only after all nine immutable completions pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

PROTOCOL_ID = "desired-results-final-test-v1"
PROTOCOL_REVISION = 1
CALIBRATION_PROTOCOL_ID = "desired-results-bounded-rerun-v1"
CALIBRATION_PROTOCOL_REVISION = 1
PRIOR_DEFINITIONS_SHA256 = "d9c8c9cefd107c86835cf486bf673ea62ecbe2f4b648ed82992d66fcc3bb5858"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_SLUG = "meta-llama__Llama-3.2-1B-Instruct"
BENCHMARK = "winogrande"
TARGET_ID = f"steering_effectiveness_initial:{MODEL_SLUG}:{BENCHMARK}"
ACTIVATION_REVISION = "8c01dd5342f5b13c6d62eca9c343cd9714ec2e9b"
FULL_SUPPORT_SHA256 = "04aa45f7726936eea778be76eada31746b97ffcbb712dabfd3fa628d30142c7c"
PAIR_TEXT_SHA256 = "24511b10962c2ebfba4553217b9619949dfe623a64ac01be685093f2fdfbdeae"
METHODS = ("caa", "grom", "mlp", "nurt", "ostrze", "tecza", "tetno", "wicher")
ARMS = ("baseline",) + METHODS
FORMATS = (
    "chat_first", "chat_last", "chat_mean", "chat_max_norm", "chat_weighted",
    "mc_balanced", "role_play",
)
CODE_REVISION = "a79242ac146502a9f2b2b8c10c8af2eff82e86f4"
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_SELECTION_KEYS = {
    "score", "scores", "best_score", "best_validation_score", "validation_score",
    "validation_summary", "validation_responses", "responses", "trials", "trial_scores",
    "validation_pair_ids", "selection_pair_ids", "test_pair_ids",
}


class FinalTestError(RuntimeError):
    """The immutable final-test contract is incomplete or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FinalTestError(f"value is not canonical JSON: {exc}") from exc


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite number {token}")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FinalTestError(f"cannot read strict JSON {path}: {exc}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                         allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_exact_keys(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(keys)
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise FinalTestError(f"{label} keys differ: expected {sorted(expected)!r}, got {actual!r}")
    return value


def _artifact_ref(value: Any, label: str, base: Path) -> Dict[str, Any]:
    """Validate an immutable local+remote artifact identity without deserializing it."""
    _require_exact_keys(value, {"path", "uri", "sha256", "generation"}, label)
    path = Path(value["path"])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FinalTestError(f"{label} local artifact does not exist: {path}")
    if not isinstance(value["uri"], str) or not value["uri"].startswith("gs://"):
        raise FinalTestError(f"{label}.uri must be gs://")
    if not isinstance(value["generation"], str) or not value["generation"].isdigit():
        raise FinalTestError(f"{label}.generation must be a decimal string")
    if not isinstance(value["sha256"], str) or not HEX64.fullmatch(value["sha256"]):
        raise FinalTestError(f"{label}.sha256 must be lowercase SHA-256")
    if _file_sha256(path) != value["sha256"]:
        raise FinalTestError(f"{label} local bytes do not match declared SHA-256")
    return {"uri": value["uri"], "sha256": value["sha256"],
            "generation": value["generation"]}


def _walk_forbidden(value: Any, label: str = "selection") -> None:
    if isinstance(value, dict):
        bad = set(value) & _FORBIDDEN_SELECTION_KEYS
        if bad:
            raise FinalTestError(f"{label} exposes forbidden score/validation fields: {sorted(bad)!r}")
        for key, child in value.items():
            _walk_forbidden(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{label}[{index}]")


def _load_inventory(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FinalTestError(f"inventory does not exist: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        targets = connection.execute(
            "SELECT * FROM prepared_targets WHERE model_name=? AND benchmark=?", (MODEL, BENCHMARK)
        ).fetchall()
        if len(targets) != 1:
            raise FinalTestError(f"expected one prepared target, found {len(targets)}")
        target = targets[0]
        support = connection.execute(
            "SELECT pair_id, stable_id, split_name FROM prepared_target_support "
            "WHERE target_id=? ORDER BY pair_id", (target["target_id"],)
        ).fetchall()
    except sqlite3.Error as exc:
        raise FinalTestError(f"invalid inventory schema: {exc}") from exc
    finally:
        connection.close()
    expected_target = {
        "target_id": TARGET_ID, "model_slug": MODEL_SLUG, "activation_revision": ACTIVATION_REVISION,
        "pair_count": 500, "layer_count": 16, "format_count": 7, "support_hash": FULL_SUPPORT_SHA256,
        "pair_text_hash": PAIR_TEXT_SHA256, "no_submission": 1,
    }
    for key, expected in expected_target.items():
        if target[key] != expected:
            raise FinalTestError(f"inventory target {key} differs from frozen identity")
    if len(support) != 500:
        raise FinalTestError("inventory support must contain exactly 500 rows")
    records: list[Dict[str, Any]] = []
    pair_ids: set[int] = set()
    stable_ids: set[str] = set()
    splits = {"train": [], "validation": [], "test": []}
    for row in support:
        pair_id, stable_id, split = row["pair_id"], row["stable_id"], row["split_name"]
        if type(pair_id) is not int or not isinstance(stable_id, str) or not stable_id or split not in splits:
            raise FinalTestError("inventory contains malformed support identity")
        if pair_id in pair_ids or stable_id in stable_ids:
            raise FinalTestError("inventory pair_id and stable_id must both be unique")
        pair_ids.add(pair_id); stable_ids.add(stable_id)
        item = {"pair_id": pair_id, "stable_id": stable_id}
        records.append({**item, "split": split})
        splits[split].append(item)
    if {name: len(rows) for name, rows in splits.items()} != {"train": 300, "validation": 100, "test": 100}:
        raise FinalTestError("inventory split counts must be train=300 validation=100 test=100")
    return {
        "target": {"model": MODEL, "model_slug": MODEL_SLUG, "benchmark": BENCHMARK,
                   "target_id": TARGET_ID, "optimization_run_id": "primary"},
        "identity": {"pair_text_sha256": PAIR_TEXT_SHA256,
                     "full_support_sha256": FULL_SUPPORT_SHA256,
                     "split_assignment_sha256": _canonical_json_sha256(records),
                     "train_support_sha256": _canonical_json_sha256(splits["train"]),
                     "test_support_sha256": _canonical_json_sha256(splits["test"])},
        "train": splits["train"], "test": splits["test"],
    }


def _load_calibration_index(path: Path, expected_generation: str,
                            index_uri: str | None = None) -> Dict[str, Any]:
    """Load only the index and score-free selection projections; hash all other artifacts."""
    if not isinstance(expected_generation, str) or not expected_generation.isdigit():
        raise FinalTestError("calibration index generation must be a decimal string")
    if index_uri is not None and (not isinstance(index_uri, str) or not index_uri.startswith("gs://")):
        raise FinalTestError("calibration index URI must be immutable gs:// identity")
    index_sha = _file_sha256(path)
    raw = _read_json(path)
    required = {"schema_version", "protocol", "target", "revisions", "input_identity",
                "extraction_strategies", "trials_per_method", "test_evaluations", "methods"}
    _require_exact_keys(raw, required, "calibration index")
    protocol = raw["protocol"]
    if protocol != {"id": CALIBRATION_PROTOCOL_ID, "revision": CALIBRATION_PROTOCOL_REVISION,
                    "prior_definitions_sha256": PRIOR_DEFINITIONS_SHA256}:
        raise FinalTestError("calibration protocol identity differs from frozen bounded rerun")
    if raw["target"] != {"model": MODEL, "benchmark": BENCHMARK, "target_id": TARGET_ID}:
        raise FinalTestError("calibration index target differs")
    revisions = raw["revisions"]
    _require_exact_keys(revisions, {"model", "activation"}, "calibration revisions")
    if not HEX40.fullmatch(str(revisions["model"])) or revisions["activation"] != ACTIVATION_REVISION:
        raise FinalTestError("calibration revisions are not immutable/frozen")
    if raw["input_identity"] != {"pair_text_sha256": PAIR_TEXT_SHA256,
                                 "full_support_sha256": FULL_SUPPORT_SHA256}:
        raise FinalTestError("calibration input identity differs")
    if raw["extraction_strategies"] != list(FORMATS) or raw["trials_per_method"] != 14 or raw["test_evaluations"] != 0:
        raise FinalTestError("calibration budget/format/test contract differs")
    methods = raw["methods"]
    if not isinstance(methods, dict) or set(methods) != set(METHODS):
        raise FinalTestError("calibration index must contain exactly eight methods")
    clean_methods: Dict[str, Any] = {}
    base = path.resolve().parent
    method_keys = {"selected_config", "frozen_config", "provenance", "completion", "config_sha256"}
    for method in METHODS:
        record = _require_exact_keys(methods[method], method_keys, f"calibration method {method}")
        refs = {key: _artifact_ref(record[key], f"{method}.{key}", base)
                for key in ("selected_config", "frozen_config", "provenance", "completion")}
        selected_path = Path(record["selected_config"]["path"])
        if not selected_path.is_absolute():
            selected_path = base / selected_path
        selected = _read_json(selected_path.resolve())
        _walk_forbidden(selected)
        _require_exact_keys(selected, {"schema_version", "method", "best_params", "config_sha256"},
                            f"{method} selected config")
        if selected["schema_version"] != 1 or selected["method"] != method:
            raise FinalTestError(f"{method} selection identity differs")
        params = selected["best_params"]
        if not isinstance(params, dict) or not params:
            raise FinalTestError(f"{method} params must be a nonempty object")
        for name, value in params.items():
            if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise FinalTestError(f"{method}.{name} has unsupported parameter type")
            if isinstance(value, float) and not math.isfinite(value):
                raise FinalTestError(f"{method}.{name} is non-finite")
        params_hash = _canonical_json_sha256(params)
        if (record["config_sha256"] != params_hash or selected["config_sha256"] != params_hash or
                not HEX64.fullmatch(str(record["config_sha256"]))):
            raise FinalTestError(f"{method} config hash does not match exact selected params")
        clean_methods[method] = {"params": params, "config_sha256": params_hash, **refs}
    return {"uri": index_uri or path.resolve().as_uri(), "sha256": index_sha,
            "generation": expected_generation, "model_revision": revisions["model"],
            "methods": clean_methods}


def _validate_runtime_identity(value: Any) -> Dict[str, Any]:
    required = {"container", "python", "torch", "cuda", "driver", "gpu",
                "precision", "evaluator_version", "tokenizer_revision", "coherence"}
    _require_exact_keys(value, required, "runtime identity")
    for key in required - {"coherence"}:
        if not isinstance(value[key], str) or not value[key]:
            raise FinalTestError(f"runtime identity {key} must be a nonempty exact string")
    if not isinstance(value["coherence"], dict) or not value["coherence"]:
        raise FinalTestError("runtime coherence identity must be an exact nonempty object")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value["container"]):
        raise FinalTestError("runtime container must be a pinned sha256 image digest")
    return dict(value)


def _with_hash(value: Dict[str, Any], field: str) -> Dict[str, Any]:
    result = dict(value)
    result[field] = _canonical_json_sha256(result)
    return result


def _build_contract(inventory: Mapping[str, Any], calibration: Mapping[str, Any],
                    code_revision: str, runtime_identity: Mapping[str, Any],
                    remote_prefix: str) -> Dict[str, Any]:
    if code_revision != CODE_REVISION:
        raise FinalTestError(f"code revision must equal frozen commit {CODE_REVISION}")
    runtime = _validate_runtime_identity(runtime_identity)
    if runtime["tokenizer_revision"] != calibration["model_revision"]:
        raise FinalTestError("tokenizer revision must equal the pinned model revision")
    if not isinstance(calibration.get("uri"), str) or not calibration["uri"].startswith("gs://"):
        raise FinalTestError("calibration index must retain its immutable GCS URI")
    prefix = remote_prefix.rstrip("/") + "/"
    if not prefix.startswith("gs://") or "/final-test-v1/" not in prefix:
        raise FinalTestError("remote prefix must be the target final-test-v1 GCS prefix")
    metric = {
        "evaluator": "log_likelihoods", "primary_metric": "aggregated_metrics.acc",
        "diagnostics": ["raw_accuracy", "correct_count", "mean_confidence", "coherence_factor"],
        "expected_count": 100, "skip_or_error_policy": "refuse",
        "aggregation": "arithmetic mean over exact ordered test support",
        "constants": {"runtime": runtime, "task": BENCHMARK},
    }
    metric["metric_contract_sha256"] = _canonical_json_sha256(metric)
    contract = {
        "schema_version": 1,
        "protocol": {"id": PROTOCOL_ID, "revision": PROTOCOL_REVISION,
                     "run_class": "immutable_final_test",
                     "calibration_protocol_id": CALIBRATION_PROTOCOL_ID,
                     "calibration_protocol_revision": CALIBRATION_PROTOCOL_REVISION,
                     "prior_definitions_sha256": PRIOR_DEFINITIONS_SHA256},
        "target": dict(inventory["target"]),
        "revisions": {"code": code_revision, "model": calibration["model_revision"],
                      "activation": ACTIVATION_REVISION, "runtime": runtime},
        "input_identity": dict(inventory["identity"]),
        "split_contract": {
            "fit": {"name": "train", "count": 300, "support": list(inventory["train"])},
            "selection": {"name": None, "pair_ids": [], "reads": 0},
            "evaluation": {"name": "test", "count": 100, "support": list(inventory["test"]),
                           "evaluations_per_arm": 1},
            "validation_pair_ids_forbidden": True,
        },
        "calibration": {"index": {key: calibration[key] for key in ("uri", "sha256", "generation")},
                        "methods": dict(calibration["methods"])},
        "arms": list(ARMS),
        "baseline": {"fit": "none", "config": None, "steering_object": "forbidden",
                     "steering_hooks": "forbidden", "same_evaluator": True},
        "metric_contract": metric,
        "execution": {"arm_count": 9, "stado_gpu_jobs": 9, "max_attempts": 1,
                      "retry": "forbidden", "claim_before": ["model_load", "train_fit", "test_read"],
                      "configuration_mutation": "forbidden", "optimization": "forbidden",
                      "test_read_per_arm": 1},
        "publication": {"remote_prefix": prefix, "gcs_precondition": "ifGenerationMatch=0",
                        "completion_last": True, "partial_leaderboard": "forbidden"},
    }
    return _with_hash(contract, "contract_sha256")


def _build_manifests(contract: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    contract_hash = contract["contract_sha256"]
    common = {
        "schema_version": 1, "protocol": contract["protocol"], "target": contract["target"],
        "contract_sha256": contract_hash, "revisions": contract["revisions"],
        "input_identity": contract["input_identity"], "split_contract": contract["split_contract"],
        "metric_contract": contract["metric_contract"], "test_evaluations": 1,
    }
    prefix = contract["publication"]["remote_prefix"]
    manifests = {}
    for arm in ARMS:
        output_prefix = f"{prefix}runs/{arm}/{contract_hash}/"
        manifest = {
            **common, "arm": arm, "method": None if arm == "baseline" else arm,
            "calibration": None if arm == "baseline" else contract["calibration"]["methods"][arm],
            "claim_uri": f"{prefix}control/claims/{arm}.json", "output_prefix": output_prefix,
        }
        manifests[arm] = _with_hash(manifest, "manifest_sha256")
    return manifests


def _build_stado_jobs(contract: Mapping[str, Any], sealed_manifests: Mapping[str, Any],
                      seal_ref: Mapping[str, str]) -> list[Dict[str, Any]]:
    jobs = []
    short = contract["contract_sha256"][:12]
    runtime = contract["revisions"]["runtime"]
    for arm in ARMS:
        ref = sealed_manifests[arm]
        jobs.append({
            "name": f"desired-results-final-test-v1-{arm}-{short}",
            "arm": arm, "accelerator": "nvidia-rtx-pro-6000", "maxAttempts": 1,
            "retry": "forbidden", "environment": {"PYTHONPATH": "."},
            "identity": {"code": contract["revisions"]["code"], "model": contract["revisions"]["model"],
                         "activation": contract["revisions"]["activation"], "runtime": runtime,
                         "metric_contract_sha256": contract["metric_contract"]["metric_contract_sha256"]},
            "inputs": {"manifest": dict(ref), "seal": dict(seal_ref)},
            "command": ["python", "scripts/steering/desired_results_final_test_worker.py",
                        "--manifest", ref["uri"], "--seal", seal_ref["uri"],
                        "--remote-prefix", contract["publication"]["remote_prefix"], "--device", "cuda"],
        })
    return jobs

def _stado_plan(args: argparse.Namespace, store: GCSStore | None = None) -> Dict[str, Any]:
    """Re-read the immutable seal and emit the one-attempt nine-job fan-out."""
    store = store or GCSStore()
    seal_data, seal_generation = store.read(args.seal, args.seal_generation)
    seal = _strict_json_bytes(seal_data, "seal")
    if seal.get("seal_sha256") != _canonical_json_sha256(
            {key: value for key, value in seal.items() if key != "seal_sha256"}):
        raise FinalTestError("seal hash mismatch")
    if seal.get("protocol_id") != PROTOCOL_ID or seal.get("arms") != list(ARMS):
        raise FinalTestError("seal does not authorize exactly the final-test arms")
    contract = _read_ref(store, seal["contract"], "contract")
    if contract.get("contract_sha256") != seal.get("contract_sha256"):
        raise FinalTestError("seal contract reference differs from canonical contract identity")
    seal_ref = {
        "uri": args.seal, "generation": seal_generation,
        "sha256": hashlib.sha256(seal_data).hexdigest(), "size": str(len(seal_data)),
    }
    return {"stado_jobs": _build_stado_jobs(contract, seal["manifests"], seal_ref)}


class GCSStore:
    """Minimal immutable-object API; every write is a create-only CAS."""
    def __init__(self) -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise FinalTestError("google-cloud-storage is required for GCS modes") from exc
        self._client = storage.Client()

    @staticmethod
    def _parts(uri: str) -> tuple[str, str]:
        if not isinstance(uri, str) or not uri.startswith("gs://"):
            raise FinalTestError(f"not a GCS URI: {uri!r}")
        bucket, sep, name = uri[5:].partition("/")
        if not bucket or not sep or not name:
            raise FinalTestError(f"incomplete GCS URI: {uri!r}")
        return bucket, name

    def create(self, uri: str, data: bytes, content_type: str = "application/json") -> Dict[str, str]:
        bucket, name = self._parts(uri)
        blob = self._client.bucket(bucket).blob(name)
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        except Exception as exc:
            raise FinalTestError(f"create-only upload refused for {uri}: {exc}") from exc
        blob.reload()
        return {"uri": uri, "generation": str(blob.generation),
                "sha256": hashlib.sha256(data).hexdigest(), "size": str(len(data))}

    def exists(self, uri: str) -> bool:
        bucket, name = self._parts(uri)
        return bool(self._client.bucket(bucket).blob(name).exists())

    def read(self, uri: str, generation: str | None = None) -> tuple[bytes, str]:
        bucket, name = self._parts(uri)
        blob = self._client.bucket(bucket).blob(name, generation=int(generation) if generation else None)
        try:
            data = blob.download_as_bytes(if_generation_match=int(generation) if generation else None)
            blob.reload()
        except Exception as exc:
            raise FinalTestError(f"cannot read immutable object {uri}: {exc}") from exc
        observed = str(blob.generation)
        if generation is not None and observed != str(generation):
            raise FinalTestError(f"generation drift for {uri}")
        return data, observed


def _prepare(args: argparse.Namespace) -> Dict[str, Any]:
    calibration = _load_calibration_index(
        args.calibration_index.resolve(), args.index_generation, args.calibration_index_uri)
    runtime = _read_json(args.runtime_identity.resolve())
    _validate_runtime_identity(runtime)
    inventory = _load_inventory(args.inventory.resolve())
    contract = _build_contract(inventory, calibration, args.code_revision, runtime, args.remote_prefix)
    manifests = _build_manifests(contract)
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FinalTestError(f"prepare destination already exists: {destination}")
    destination.mkdir(parents=True)
    _atomic_json(destination / "contract.json", contract)
    manifest_dir = destination / "manifests"
    for arm, manifest in manifests.items():
        _atomic_json(manifest_dir / f"{arm}.json", manifest)
    bundle = {"schema_version": 1, "contract_sha256": contract["contract_sha256"],
              "contract_file": "contract.json", "manifests": {arm: f"manifests/{arm}.json" for arm in ARMS}}
    _atomic_json(destination / "bundle.json", bundle)
    return bundle


def _load_bundle(directory: Path) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    bundle = _read_json(directory / "bundle.json")
    _require_exact_keys(bundle, {"schema_version", "contract_sha256", "contract_file", "manifests"}, "bundle")
    contract = _read_json(directory / bundle["contract_file"])
    expected_hash = contract.pop("contract_sha256", None)
    contract["contract_sha256"] = expected_hash
    if expected_hash != bundle["contract_sha256"] or _canonical_json_sha256({k: v for k, v in contract.items() if k != "contract_sha256"}) != expected_hash:
        raise FinalTestError("bundle contract hash mismatch")
    if set(bundle["manifests"]) != set(ARMS):
        raise FinalTestError("bundle does not contain exactly nine manifests")
    manifests = {}
    for arm in ARMS:
        manifest = _read_json(directory / bundle["manifests"][arm])
        observed = manifest.get("manifest_sha256")
        if observed != _canonical_json_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
            raise FinalTestError(f"{arm} manifest hash mismatch")
        manifests[arm] = manifest
    return contract, manifests


def _seal(args: argparse.Namespace, store: GCSStore | None = None) -> Dict[str, Any]:
    store = store or GCSStore()
    contract, manifests = _load_bundle(args.bundle.resolve())
    prefix = contract["publication"]["remote_prefix"]
    contract_uri = f"{prefix}control/contract/{contract['contract_sha256']}.json"
    guarded = [
        f"{prefix}control/seal.json", f"{prefix}publication.json",
        f"{prefix}aggregate/{contract['contract_sha256']}/publication.json", contract_uri,
    ]
    for arm in ARMS:
        guarded.extend([
            f"{prefix}control/claims/{arm}.json",
            f"{prefix}runs/{arm}/{contract['contract_sha256']}/completion.json",
            f"{prefix}control/manifests/{arm}/{manifests[arm]['manifest_sha256']}.json",
        ])
    if any(store.exists(uri) for uri in guarded):
        raise FinalTestError("seal/claim/completion/aggregate/control object already exists; rerun forbidden")
    contract_ref = store.create(contract_uri, _canonical_bytes(contract))
    refs = {}
    for arm in ARMS:
        manifest = manifests[arm]
        uri = f"{prefix}control/manifests/{arm}/{manifest['manifest_sha256']}.json"
        refs[arm] = store.create(uri, _canonical_bytes(manifest))
    seal = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "contract": contract_ref, "manifests": refs, "arms": list(ARMS),
        "contract_sha256": contract["contract_sha256"],
        "runtime_identity": contract["revisions"]["runtime"],
        "metric_contract_sha256": contract["metric_contract"]["metric_contract_sha256"],
    }
    seal["seal_sha256"] = _canonical_json_sha256(seal)
    seal_uri = f"{prefix}control/seal.json"
    seal_ref = store.create(seal_uri, _canonical_bytes(seal))
    jobs = _build_stado_jobs(contract, refs, seal_ref)
    output = {"seal": seal_ref, "contract": contract_ref, "manifests": refs, "stado_jobs": jobs}
    if args.output:
        _atomic_json(args.output.resolve(), output)
    return output


def _strict_json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("ascii"), parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite {token}")))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise FinalTestError(f"invalid JSON bytes for {label}: {exc}") from exc


def _read_ref_bytes(store: GCSStore, ref: Mapping[str, Any], label: str) -> bytes:
    _require_exact_keys(ref, {"uri", "generation", "sha256", "size"}, label)
    data, _ = store.read(ref["uri"], ref["generation"])
    if hashlib.sha256(data).hexdigest() != ref["sha256"] or len(data) != int(ref["size"]):
        raise FinalTestError(f"{label} immutable bytes differ")
    return data


def _read_ref(store: GCSStore, ref: Mapping[str, Any], label: str) -> Any:
    return _strict_json_bytes(_read_ref_bytes(store, ref, label), label)


def _validate_completion(arm: str, completion: Mapping[str, Any], contract: Mapping[str, Any],
                         manifest_ref: Mapping[str, Any], store: GCSStore) -> Dict[str, Any]:
    required = {"schema_version", "arm", "contract_sha256", "manifest_sha256", "manifest_generation",
                "metric_contract_sha256", "artifacts", "completion_sha256"}
    _require_exact_keys(completion, required, f"{arm} completion")
    claimed = completion["completion_sha256"]
    if claimed != _canonical_json_sha256({k: v for k, v in completion.items() if k != "completion_sha256"}):
        raise FinalTestError(f"{arm} completion hash mismatch")
    manifest = _read_ref(store, manifest_ref, f"{arm} manifest")
    if (not isinstance(manifest, dict) or manifest.get("arm") != arm or
            manifest.get("contract_sha256") != contract["contract_sha256"] or
            manifest.get("manifest_sha256") != _canonical_json_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"})):
        raise FinalTestError(f"{arm} sealed manifest identity differs")
    if (completion["arm"] != arm or completion["contract_sha256"] != contract["contract_sha256"] or
            completion["manifest_sha256"] != manifest["manifest_sha256"] or
            completion["manifest_generation"] != manifest_ref["generation"] or
            completion["metric_contract_sha256"] != contract["metric_contract"]["metric_contract_sha256"]):
        raise FinalTestError(f"{arm} completion identity mismatch")
    artifacts = completion["artifacts"]
    expected_files = {"result.json", "test_predictions.jsonl", "scores.json", "responses.json", "provenance.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_files:
        raise FinalTestError(f"{arm} completion artifact set differs")
    output_prefix = (contract["publication"]["remote_prefix"] +
                     f"runs/{arm}/{contract['contract_sha256']}/")
    for name, ref in artifacts.items():
        if not isinstance(ref, dict) or ref.get("uri") != output_prefix + name:
            raise FinalTestError(f"{arm}/{name} artifact URI differs from sealed output")
    values = {name: _read_ref(store, ref, f"{arm}/{name}")
              for name, ref in artifacts.items() if name != "test_predictions.jsonl"}
    result = values["result.json"]
    if not isinstance(result, dict) or result.get("arm") != arm:
        raise FinalTestError(f"{arm} result identity differs")
    scores = values["scores.json"]
    responses = values["responses.json"]
    if (not isinstance(scores, dict) or scores.get("evaluator_used") != "log_likelihoods" or
            scores.get("num_total") != 100 or scores.get("num_evaluated") != 100 or
            scores.get("num_model_required") != 0 or
            not isinstance(scores.get("evaluations"), list) or len(scores["evaluations"]) != 100 or
            not isinstance(responses, dict) or not isinstance(responses.get("responses"), list) or
            len(responses["responses"]) != 100):
        raise FinalTestError(f"{arm} evaluator artifacts are not complete log-likelihood outputs")
    for key in ("primary_metric", "raw_accuracy"):
        if isinstance(result.get(key), bool) or not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]):
            raise FinalTestError(f"{arm} result {key} is not finite")
    if result.get("num_total") != 100 or result.get("num_evaluated") != 100 or result.get("num_errors") != 0 or result.get("num_skipped") != 0:
        raise FinalTestError(f"{arm} result is not a complete 100/100 evaluation")
    expected_support = contract["split_contract"]["evaluation"]["support"]
    expected_order_hash = _canonical_json_sha256([row["pair_id"] for row in expected_support])
    if result.get("ordered_test_ids_sha256") != expected_order_hash:
        raise FinalTestError(f"{arm} ordered test identity hash differs")
    if result.get("test_support_sha256") != _canonical_json_sha256(expected_support):
        raise FinalTestError(f"{arm} test support hash differs")
    prediction_data = _read_ref_bytes(store, artifacts["test_predictions.jsonl"], f"{arm}/test_predictions.jsonl")
    try:
        lines = prediction_data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise FinalTestError(f"{arm} predictions are not ASCII JSONL") from exc
    if any(not line for line in lines):
        raise FinalTestError(f"{arm} predictions contain a blank JSONL row")
    predictions = [_strict_json_bytes(line.encode("ascii"), f"{arm} prediction {index}")
                   for index, line in enumerate(lines, 1)]
    if len(predictions) != 100:
        raise FinalTestError(f"{arm} predictions do not contain exactly 100 rows")
    correct_count = 0
    for expected, prediction, evaluation in zip(expected_support, predictions, scores["evaluations"]):
        _require_exact_keys(prediction, {"pair_id", "stable_id", "correct", "confidence", "evaluation"},
                            f"{arm} prediction")
        confidence = prediction["confidence"]
        outcome = evaluation.get("evaluation") if isinstance(evaluation, dict) else None
        if (prediction["pair_id"] != expected["pair_id"] or prediction["stable_id"] != expected["stable_id"] or
                type(prediction["correct"]) is not bool or isinstance(confidence, bool) or
                not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or
                not isinstance(outcome, dict) or "error" in evaluation or "error" in outcome or
                outcome != prediction["evaluation"] or outcome.get("correct") != prediction["correct"] or
                outcome.get("confidence") != confidence):
            raise FinalTestError(f"{arm} predictions differ from ordered sealed evaluator output")
        correct_count += int(prediction["correct"])
    acc = scores.get("aggregated_metrics", {}).get("acc")
    if (result.get("correct_count") != correct_count or result.get("raw_accuracy") != correct_count / 100.0 or
            isinstance(acc, bool) or not isinstance(acc, (int, float)) or not math.isfinite(acc) or
            result.get("primary_metric") != float(acc)):
        raise FinalTestError(f"{arm} normalized metrics differ from evaluator artifacts")
    return result

def _check_comparability(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    fields = ("metric_contract_sha256", "test_support_sha256", "ordered_test_ids_sha256",
              "pair_text_sha256", "model_revision", "tokenizer_revision", "code_revision",
              "runtime_identity_sha256", "evaluator", "evaluator_version", "evaluation_mode",
              "sample_count", "aggregation")
    shared = {}
    first = results[ARMS[0]]
    for field in fields:
        value = first.get(field)
        if value is None or any(results[arm].get(field) != value for arm in ARMS):
            raise FinalTestError(f"final-test arms are not comparable on {field}")
        shared[field] = value
    return {"schema_version": 1, "comparable": True, "arms": list(ARMS), "shared": shared}


def _finalize(args: argparse.Namespace, store: GCSStore | None = None) -> Dict[str, Any]:
    store = store or GCSStore()
    seal_data, seal_generation = store.read(args.seal, args.seal_generation)
    seal = _strict_json_bytes(seal_data, "seal")
    if seal.get("seal_sha256") != _canonical_json_sha256({k: v for k, v in seal.items() if k != "seal_sha256"}):
        raise FinalTestError("seal hash mismatch")
    contract = _read_ref(store, seal["contract"], "contract")
    if seal.get("protocol_id") != PROTOCOL_ID or seal.get("arms") != list(ARMS):
        raise FinalTestError("seal does not cover exactly all final-test arms")
    prefix = contract["publication"]["remote_prefix"]
    results = {}
    if contract.get("contract_sha256") != seal.get("contract_sha256"):
        raise FinalTestError("sealed contract identity differs")
    if (store.exists(prefix + "publication.json") or
            store.exists(prefix + f"aggregate/{contract['contract_sha256']}/publication.json")):
        raise FinalTestError("final-test publication already exists; rerun forbidden")
    completions = {}
    for arm in ARMS:
        uri = f"{prefix}runs/{arm}/{contract['contract_sha256']}/completion.json"
        data, generation = store.read(uri)
        completion = _strict_json_bytes(data, f"{arm} completion")
        results[arm] = _validate_completion(arm, completion, contract, seal["manifests"][arm], store)
        completions[arm] = {"uri": uri, "generation": generation,
                            "sha256": hashlib.sha256(data).hexdigest(), "size": str(len(data))}
    comparability = _check_comparability(results)
    ordered = sorted(ARMS, key=lambda arm: (-float(results[arm]["primary_metric"]), arm))
    baseline = float(results["baseline"]["primary_metric"])
    leaderboard = {"schema_version": 1, "metric": "coherence_adjusted_accuracy",
                   "single_fixed_test_no_statistical_claims": True,
                   "rows": [{"rank": rank, "arm": arm,
                              "primary_metric": results[arm]["primary_metric"],
                              "raw_accuracy": results[arm]["raw_accuracy"],
                              "baseline_delta": float(results[arm]["primary_metric"]) - baseline}
                             for rank, arm in enumerate(ordered, 1)]}
    aggregate_prefix = f"{prefix}aggregate/{contract['contract_sha256']}/"
    payloads = {
        "arm-results.json": {"schema_version": 1, "arms": results},
        "leaderboard.json": leaderboard, "comparability.json": comparability,
        "provenance.json": {"schema_version": 1, "contract_sha256": contract["contract_sha256"],
                            "seal": {"uri": args.seal, "generation": seal_generation,
                                     "sha256": hashlib.sha256(seal_data).hexdigest()},
                            "completions": completions},
    }
    refs = {name: store.create(aggregate_prefix + name, _canonical_bytes(value))
            for name, value in payloads.items()}
    receipt = {"schema_version": 1, "contract_sha256": contract["contract_sha256"],
               "aggregate_objects": refs, "complete_arms": list(ARMS)}
    receipt["publication_sha256"] = _canonical_json_sha256(receipt)
    receipt_ref = store.create(aggregate_prefix + "publication.json", _canonical_bytes(receipt))
    pointer = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
               "contract_sha256": contract["contract_sha256"], "aggregate": receipt_ref}
    pointer["publication_pointer_sha256"] = _canonical_json_sha256(pointer)
    pointer_ref = store.create(prefix + "publication.json", _canonical_bytes(pointer))
    return {"publication": pointer_ref, "leaderboard": leaderboard}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--inventory", type=Path, required=True)
    prepare.add_argument("--calibration-index", type=Path, required=True)
    prepare.add_argument("--index-generation", required=True)
    prepare.add_argument("--code-revision", required=True)
    prepare.add_argument("--runtime-identity", type=Path, required=True)
    prepare.add_argument("--remote-prefix", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    plan = sub.add_parser("stado-plan")
    plan.add_argument("--seal", required=True)
    plan.add_argument("--seal-generation", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--calibration-index-uri", required=True)
    seal.add_argument("--output", type=Path)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--seal", required=True)
    finalize.add_argument("--seal-generation", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "prepare":
            result = _prepare(args)
        elif args.mode == "seal":
            result = _seal(args)
        elif args.mode == "stado-plan":
            result = _stado_plan(args)
        else:
            result = _finalize(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (FinalTestError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"desired-results final test refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
