#!/usr/bin/env python3
"""Fail-closed preflight execution gate for one desired-results method run.

The gate reads the prepared inventory and atomically writes a deterministic
execution manifest. It never loads a model or activations, runs optimization,
submits compute, or writes the shared canonical result leaf.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROTOCOL_ID = "steering_effectiveness_initial"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BENCHMARK = "winogrande"
ELIGIBLE_METHODS = (
    "baseline",
    "caa",
    "ostrze",
    "mlp",
    "tecza",
    "tetno",
    "grom",
    "nurt",
    "wicher",
)
DEFERRED_METHODS = ("szlak", "przelom")
EXTRACTION_COMPONENT = "residual_stream"
EXTRACTION_STRATEGIES = (
    "chat_first",
    "chat_last",
    "chat_mean",
    "chat_max_norm",
    "chat_weighted",
    "mc_balanced",
    "role_play",
)
SPLITS = ("train", "validation", "test")
PURPOSES = ("preflight", "calibration")
MODES = {"preflight": "preflight", "calibration": "calibration"}
DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[3]
    / ".work/results_scope/desired_results_state_v1/result_inventory.sqlite"
)


class PolicyError(RuntimeError):
    """An inventory or execution request violates the frozen policy."""


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_one(connection: sqlite3.Connection, query: str, values: Sequence[Any]) -> sqlite3.Row:
    rows = connection.execute(query, values).fetchall()
    if len(rows) != 1:
        raise PolicyError(f"expected exactly one inventory row, found {len(rows)}")
    return rows[0]


def _validate_output_prefix(prefix: str, model_slug: str, method: str, run_id: str) -> str:
    expected = f"runs/{PROTOCOL_ID}/{model_slug}/{BENCHMARK}/{method}/{run_id}/"
    if prefix != expected:
        raise PolicyError(f"inventory staging_prefix is not the isolated expected prefix: {prefix!r}")
    parts = Path(prefix).parts
    if not parts or parts[0] != "runs" or "results" in parts or ".." in parts:
        raise PolicyError("method output prefix could reach the shared canonical result leaf")
    return prefix


def _load_job(
    inventory: Path,
    model: str,
    benchmark: str,
    method: str,
    run_id: str,
    model_revision: str,
    purpose: str = "preflight",
) -> Dict[str, Any]:
    if purpose not in PURPOSES:
        raise PolicyError(f"unknown manifest purpose {purpose!r}")
    if model != MODEL_NAME or benchmark != BENCHMARK:
        raise PolicyError(f"this worker is pinned to {MODEL_NAME} x {BENCHMARK}")
    if method in DEFERRED_METHODS:
        raise PolicyError(f"method {method} is deferred_special_case and cannot be executed")
    if method not in ELIGIBLE_METHODS:
        raise PolicyError(f"method {method!r} is outside the frozen eligible method scope")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise PolicyError("model_revision must be an immutable 40-character lowercase commit SHA")
    if not inventory.is_file():
        raise PolicyError(f"inventory does not exist: {inventory}")

    uri = f"file:{inventory.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        target = _read_one(
            connection,
            "SELECT * FROM prepared_targets WHERE model_name=? AND benchmark=?",
            (model, benchmark),
        )
        run = _read_one(
            connection,
            "SELECT * FROM prepared_method_runs "
            "WHERE target_id=? AND method=? AND optimization_run_id=?",
            (target["target_id"], method, run_id),
        )
        method_row = _read_one(
            connection,
            "SELECT * FROM methods WHERE method=?",
            (method,),
        )
        support = connection.execute(
            "SELECT pair_id, stable_id, split_name FROM prepared_target_support "
            "WHERE target_id=? ORDER BY pair_id",
            (target["target_id"],),
        ).fetchall()
    finally:
        connection.close()

    if target["preflight_status"] != "ready_metadata_and_identity":
        raise PolicyError(f"target preflight is not ready: {target['preflight_status']}")
    if int(target["no_submission"]) != 1:
        raise PolicyError("prepared target is not marked no_submission")
    if int(target["format_count"]) != len(EXTRACTION_STRATEGIES):
        raise PolicyError("inventory format count does not match the frozen seven-format scope")
    if run["eligibility"] != "eligible":
        reason = run["reason"] or "no reason recorded"
        raise PolicyError(f"method status is {run['eligibility']}: {reason}")
    if method_row["required_data"] not in ("no_steering", "residual_stream", "residual_stream_or_sequential_forward"):
        raise PolicyError("method inventory requires a component outside the residual-stream policy")
    if not support or len(support) != int(target["pair_count"]):
        raise PolicyError("prepared support size does not match target pair_count")

    split_pair_ids = {name: [] for name in SPLITS}  # type: Dict[str, List[int]]
    support_records = []
    stable_ids = set()
    pair_ids = set()
    for row in support:
        split = row["split_name"]
        pair_id = row["pair_id"]
        stable_id = row["stable_id"]
        if split not in split_pair_ids:
            raise PolicyError(f"unknown split {split!r}")
        if pair_id is None or stable_id is None:
            raise PolicyError("support contains a missing pair_id or stable_id")
        if pair_id in pair_ids or stable_id in stable_ids:
            raise PolicyError("support contains duplicate pair_id or stable_id")
        pair_ids.add(pair_id)
        stable_ids.add(stable_id)
        split_pair_ids[split].append(int(pair_id))
        support_records.append({"pair_id": int(pair_id), "stable_id": stable_id, "split": split})
    if any(not split_pair_ids[name] for name in SPLITS):
        raise PolicyError("train, validation, and test must all be non-empty")
    if set().union(*(set(ids) for ids in split_pair_ids.values())) != pair_ids:
        raise PolicyError("split union does not equal prepared support")

    output_prefix = _validate_output_prefix(
        run["staging_prefix"], target["model_slug"], method, run_id
    )
    if purpose == "calibration":
        exposed_splits = ("train", "validation")
        split_contract = {
            "counts": {name: len(split_pair_ids[name]) for name in exposed_splits},
            "pair_ids": {name: split_pair_ids[name] for name in exposed_splits},
            "hpo_reads": ["train"],
            "selection_split": "validation",
            "final_fit": ["train"],
            "test_evaluations": 0,
        }
        mode_contracts = {
            "hpo": {
                "strict_loader_pair_ids": "train_plus_validation_only",
                "objective_reports": "validation_only",
                "writes_under": f"{output_prefix}hpo/",
                "required_output": "frozen_config.json",
            },
        }
    else:
        split_contract = {
            "counts": {name: len(split_pair_ids[name]) for name in SPLITS},
            "pair_ids": split_pair_ids,
            "hpo_reads": ["train", "validation"],
            "selection_split": "validation",
            "final_fit": ["train", "validation"],
            "final_test_reads": ["test"],
            "test_evaluations": 1,
        }
        mode_contracts = {
            "hpo": {
                "strict_loader_pair_ids": "train_plus_validation_only",
                "objective_reports": "validation_only",
                "writes_under": f"{output_prefix}hpo/",
                "required_output": "frozen_config.json",
            },
            "final_test": {
                "requires": f"{output_prefix}hpo/frozen_config.json",
                "strict_loader_pair_ids": "test_only",
                "configuration_mutation": "forbidden",
                "evaluations": 1,
                "writes_under": f"{output_prefix}final_test/",
            },
        }
    return {
        "schema_version": 1,
        "purpose": purpose,
        "job_unit": {
            "model": model,
            "benchmark": benchmark,
            "method": method,
            "optimization_run_id": run_id,
            "job_key": run["job_key"],
            "target_id": target["target_id"],
        },
        "revisions": {
            "model": model_revision,
            "activation": target["activation_revision"],
        },
        "input_identity": {
            "join_key": ["benchmark", "pair_id"],
            "pair_text_hash": target["pair_text_hash"],
            "support_hash": target["support_hash"],
            "split_assignment_hash": _sha256_json(support_records),
        },
        "split": split_contract,
        "activation_search_scope": {
            "extraction_component": EXTRACTION_COMPONENT,
            "extraction_strategies": list(EXTRACTION_STRATEGIES),
            "layers": list(range(1, int(target["layer_count"]) + 1)),
        },
        "saved_activation_policy": {
            "loader": "wisent.core.reading.modules.utilities.data.enriched_builder.build_enriched_from_hf_strict",
            "complete_marker_required": True,
            "automatic_regeneration": "forbidden",
            "fallback": "forbidden",
            "positional_join": "forbidden",
        },
        "mode_contracts": mode_contracts,
        "method_status": "eligible",
        "output_prefix": output_prefix,
        "write_policy": {
            "atomic_method_staging": True,
            "shared_canonical_result_leaf": "forbidden",
            "canonical_leaf_writer": "finalizer_only",
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _default_manifest_path(
    inventory: Path, manifest: Mapping[str, Any], purpose: str
) -> Path:
    unit = manifest["job_unit"]
    model_slug = unit["model"].replace("/", "__")
    directory = "calibration_manifests" if purpose == "calibration" else "preflight_manifests"
    return (
        inventory.parent
        / directory
        / model_slug
        / unit["benchmark"]
        / unit["method"]
        / unit["optimization_run_id"]
        / "manifest.json"
    )




def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", required=True, choices=ELIGIBLE_METHODS + DEFERRED_METHODS)
    parser.add_argument("--optimization-run", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--mode", choices=tuple(MODES), default="preflight")
    parser.add_argument("--manifest-out", type=Path)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        purpose = MODES[args.mode]
        manifest = _load_job(
            args.inventory.resolve(),
            args.model,
            args.benchmark,
            args.method,
            args.optimization_run,
            args.model_revision,
            purpose=purpose,
        )
        manifest["execution_mode"] = args.mode
        output = args.manifest_out or _default_manifest_path(
            args.inventory.resolve(), manifest, purpose
        )
        _atomic_json(output.resolve(), manifest)
        print(output.resolve())
        return 0
    except (PolicyError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
