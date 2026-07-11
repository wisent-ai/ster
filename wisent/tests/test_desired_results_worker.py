import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "scripts" / "steering" / "desired_results_worker.py"
SPEC = importlib.util.spec_from_file_location("desired_results_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _inventory(tmp_path):
    path = tmp_path / "inventory.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE prepared_targets (
            target_id TEXT PRIMARY KEY, model_name TEXT NOT NULL, model_slug TEXT NOT NULL,
            benchmark TEXT NOT NULL, activation_revision TEXT NOT NULL, preflight_status TEXT NOT NULL,
            pair_count INTEGER NOT NULL, layer_count INTEGER NOT NULL, format_count INTEGER NOT NULL,
            support_hash TEXT NOT NULL, pair_text_hash TEXT NOT NULL, no_submission INTEGER NOT NULL,
            preflight_notes TEXT NOT NULL
        );
        CREATE TABLE prepared_method_runs (
            job_key TEXT PRIMARY KEY, target_id TEXT NOT NULL, method TEXT NOT NULL,
            optimization_run_id TEXT NOT NULL, eligibility TEXT NOT NULL, reason TEXT NOT NULL,
            staging_prefix TEXT NOT NULL, stado_provider TEXT NOT NULL, stado_gpu_type TEXT NOT NULL,
            submission_status TEXT NOT NULL
        );
        CREATE TABLE methods (
            method TEXT PRIMARY KEY, kind TEXT NOT NULL, required_data TEXT NOT NULL,
            policy TEXT NOT NULL, ineligibility_rule TEXT NOT NULL
        );
        CREATE TABLE prepared_target_support (
            target_id TEXT NOT NULL, pair_id INTEGER NOT NULL, stable_id TEXT NOT NULL,
            split_name TEXT NOT NULL
        );
    """)
    target_id = "target-1"
    connection.execute(
        "INSERT INTO prepared_targets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (target_id, worker.MODEL_NAME, "meta-llama--llama-3.2-1b-instruct", worker.BENCHMARK,
         "activation-revision", "ready_metadata_and_identity", 6, 16, 7,
         "support-hash", "text-hash", 1, "ready"),
    )
    connection.execute(
        "INSERT INTO methods VALUES (?,?,?,?,?)",
        ("caa", "steering", "residual_stream", "required", ""),
    )
    prefix = (
        "runs/steering_effectiveness_initial/meta-llama--llama-3.2-1b-instruct/"
        "winogrande/caa/primary/"
    )
    connection.execute(
        "INSERT INTO prepared_method_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("job-1", target_id, "caa", "primary", "eligible", "", prefix,
         "local", "cpu", "prepared"),
    )
    support = [
        (target_id, 5, "stable-5", "validation"),
        (target_id, 9000, "stable-9000", "test"),
        (target_id, 4, "stable-4", "train"),
        (target_id, 2, "stable-2", "validation"),
        (target_id, 1, "stable-1", "train"),
        (target_id, 9003, "stable-9003", "test"),
    ]
    connection.executemany("INSERT INTO prepared_target_support VALUES (?,?,?,?)", support)
    connection.commit()
    connection.close()
    return path


def _load(path, *, purpose="preflight"):
    return worker._load_job(
        path,
        worker.MODEL_NAME,
        worker.BENCHMARK,
        "caa",
        "primary",
        "a" * 40,
        purpose=purpose,
    )


def test_preflight_manifest_freezes_disjoint_source_splits(tmp_path):
    inventory = _inventory(tmp_path)

    first = _load(inventory)
    second = _load(inventory)

    assert first == second
    split_ids = first["split"]["pair_ids"]
    assert split_ids == {
        "train": [1, 4],
        "validation": [2, 5],
        "test": [9000, 9003],
    }
    assert set(split_ids["train"]).isdisjoint(split_ids["validation"])
    assert set(split_ids["train"]).isdisjoint(split_ids["test"])
    assert set(split_ids["validation"]).isdisjoint(split_ids["test"])
    assert first["activation_search_scope"] == {
        "extraction_component": "residual_stream",
        "extraction_strategies": list(worker.EXTRACTION_STRATEGIES),
        "layers": list(range(1, 17)),
    }


def test_calibration_manifest_uses_train_for_fit_and_validation_for_selection(tmp_path):
    manifest = _load(_inventory(tmp_path), purpose="calibration")

    assert manifest["purpose"] == "calibration"
    assert manifest["split"] == {
        "counts": {"train": 2, "validation": 2},
        "pair_ids": {"train": [1, 4], "validation": [2, 5]},
        "hpo_reads": ["train"],
        "selection_split": "validation",
        "final_fit": ["train"],
        "test_evaluations": 0,
    }
    assert manifest["mode_contracts"]["hpo"]["strict_loader_pair_ids"] == (
        "train_plus_validation_only"
    )
    assert manifest["mode_contracts"]["hpo"]["objective_reports"] == "validation_only"
    assert "final_test" not in manifest["mode_contracts"]
    assert "test" not in manifest["split"]["pair_ids"]
    assert "test" not in manifest["split"]["counts"]


@pytest.mark.parametrize(
    ("missing_split", "replacement_split"),
    [("train", "validation"), ("validation", "train"), ("test", "train")],
)
def test_calibration_requires_every_source_inventory_split(
    tmp_path, missing_split, replacement_split
):
    inventory = _inventory(tmp_path)
    connection = sqlite3.connect(inventory)
    connection.execute(
        "UPDATE prepared_target_support SET split_name=? WHERE split_name=?",
        (replacement_split, missing_split),
    )
    connection.commit()
    connection.close()

    with pytest.raises(
        worker.PolicyError, match="train, validation, and test must all be non-empty"
    ):
        _load(inventory, purpose="calibration")


def test_calibration_cli_materializes_an_artifact_without_test_partition_data(
    tmp_path,
):
    inventory = _inventory(tmp_path)
    output = tmp_path / "calibration-manifest.json"

    exit_code = worker.main(
        [
            "--inventory",
            str(inventory),
            "--model",
            worker.MODEL_NAME,
            "--benchmark",
            worker.BENCHMARK,
            "--method",
            "caa",
            "--optimization-run",
            "primary",
            "--model-revision",
            "a" * 40,
            "--mode",
            "calibration",
            "--manifest-out",
            str(output),
        ]
    )
    assert exit_code == 0

    serialized = output.read_text(encoding="utf-8")
    artifact = json.loads(serialized)
    assert artifact["purpose"] == "calibration"
    assert artifact["execution_mode"] == "calibration"
    assert artifact["split"]["pair_ids"] == {
        "train": [1, 4],
        "validation": [2, 5],
    }
    assert artifact["split"]["counts"] == {"train": 2, "validation": 2}
    assert artifact["split"]["hpo_reads"] == ["train"]
    assert artifact["split"]["selection_split"] == "validation"
    assert artifact["split"]["final_fit"] == ["train"]
    assert artifact["split"]["test_evaluations"] == 0
    assert '"test"' not in serialized
    assert '"final_test"' not in serialized
    assert "9000" not in serialized
    assert "9003" not in serialized


@pytest.mark.parametrize("purpose", ["preflight", "calibration"])
@pytest.mark.parametrize("method", worker.DEFERRED_METHODS)
def test_gate_rejects_deferred_methods_for_every_purpose(tmp_path, method, purpose):
    with pytest.raises(worker.PolicyError, match="deferred_special_case"):
        worker._load_job(
            tmp_path / "unused.sqlite",
            worker.MODEL_NAME,
            worker.BENCHMARK,
            method,
            "primary",
            "a" * 40,
            purpose=purpose,
        )
