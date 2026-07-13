import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEERING = ROOT / "scripts" / "steering"
TARGET_SPEC = importlib.util.spec_from_file_location(
    "desired_results_target", STEERING / "desired_results_target.py"
)
targets = importlib.util.module_from_spec(TARGET_SPEC)
sys.modules["desired_results_target"] = targets
TARGET_SPEC.loader.exec_module(targets)
INVENTORY_SPEC = importlib.util.spec_from_file_location(
    "desired_results_inventory", STEERING / "desired_results_inventory.py"
)
inventory = importlib.util.module_from_spec(INVENTORY_SPEC)
INVENTORY_SPEC.loader.exec_module(inventory)


MODEL_ROWS = (
    ("org__small", "org/small", 3),
    ("other__wide", "other/wide", 5),
    ("third__medium", "third/medium", 4),
    ("fourth__deep", "fourth/deep", 6),
)
BENCHMARKS = tuple(f"bench_{index:03d}" for index in range(375))
BLOCKED_BENCHMARKS = frozenset(BENCHMARKS[-4:])
PROTOCOL = "steering_effectiveness_v1"
FINALIZED_KEY = ("org__small", "bench_000")
CANDIDATE_KEYS = {
    ("org__small", "bench_001"),
    ("other__wide", "bench_002"),
}
PARTIAL_KEY = ("other__wide", "bench_003")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_pairs(benchmark):
    return 6 if int(benchmark.removeprefix("bench_")) % 2 == 0 else 9


def _create_source(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE models(model_slug TEXT PRIMARY KEY, model_name TEXT NOT NULL, layer_count INTEGER NOT NULL);
        CREATE TABLE result_targets(
            protocol_id TEXT NOT NULL,
            result_id TEXT PRIMARY KEY,
            model_slug TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            expected_pairs INTEGER NOT NULL
        );
        CREATE TABLE blocked_targets(result_id TEXT PRIMARY KEY, reason_code TEXT NOT NULL, details TEXT NOT NULL);
        CREATE TABLE methods(method TEXT PRIMARY KEY);
        CREATE TABLE prepared_targets(
            target_id TEXT PRIMARY KEY,
            model_slug TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            support_hash TEXT NOT NULL
        );
        CREATE TABLE prepared_target_support(
            target_id TEXT NOT NULL,
            pair_id INTEGER NOT NULL,
            stable_id TEXT NOT NULL,
            split_name TEXT NOT NULL
        );
        CREATE TABLE prepared_method_runs(
            job_key TEXT PRIMARY KEY,
            target_id TEXT NOT NULL,
            method TEXT NOT NULL,
            optimization_run_id TEXT NOT NULL,
            staging_prefix TEXT NOT NULL,
            source_revision TEXT NOT NULL
        );
        """
    )
    connection.executemany("INSERT INTO models VALUES (?,?,?)", MODEL_ROWS)
    connection.executemany("INSERT INTO methods VALUES (?)", [("caa",), ("grom",)])
    targets_to_insert = []
    blocked_to_insert = []
    for model_slug, _, _ in MODEL_ROWS:
        for benchmark in BENCHMARKS:
            result_id = targets.result_id(PROTOCOL, model_slug, benchmark)
            targets_to_insert.append(
                (PROTOCOL, result_id, model_slug, benchmark, _expected_pairs(benchmark))
            )
            if benchmark in BLOCKED_BENCHMARKS:
                blocked_to_insert.append(
                    (result_id, "activation_source_blocked", "synthetic unavailable labels")
                )
    connection.executemany("INSERT INTO result_targets VALUES (?,?,?,?,?)", targets_to_insert)
    connection.executemany("INSERT INTO blocked_targets VALUES (?,?,?)", blocked_to_insert)

    source_target_id = "legacy-finalized-target"
    connection.execute(
        "INSERT INTO prepared_targets VALUES (?,?,?,?)",
        (source_target_id, FINALIZED_KEY[0], FINALIZED_KEY[1], "9" * 64),
    )
    split_by_pair = ("train", "train", "train", "validation", "validation", "test")
    connection.executemany(
        "INSERT INTO prepared_target_support VALUES (?,?,?,?)",
        [
            (source_target_id, pair_id, f"stable-final-{pair_id}", split_name)
            for pair_id, split_name in enumerate(split_by_pair)
        ],
    )
    connection.execute(
        "INSERT INTO prepared_method_runs VALUES (?,?,?,?,?,?)",
        (
            "legacy-job",
            source_target_id,
            "caa",
            "primary",
            "runs/org__small/bench_000/caa/primary",
            "source-revision-1",
        ),
    )
    connection.commit()
    connection.close()


def _activation_record(model_slug, benchmark, layers, *, partial=False):
    counts = {strategy: layers for strategy in targets.STRATEGIES}
    if partial:
        counts["mc_balanced"] = layers - 1
    return {
        "tier": "activations",
        "model": model_slug,
        "bench": benchmark,
        "layers": counts,
        "n_pairs": _expected_pairs(benchmark) - int(partial),
        "grouped": False,
    }


def _create_inputs(tmp_path):
    source = tmp_path / "source-v1.sqlite"
    cache = tmp_path / "synthetic-activation-cache.jsonl"
    execution = tmp_path / "execution-v3.json"
    _create_source(source)
    layer_by_model = {slug: layers for slug, _, layers in MODEL_ROWS}
    records = [
        _activation_record(*FINALIZED_KEY, layer_by_model[FINALIZED_KEY[0]]),
        *[
            _activation_record(model, benchmark, layer_by_model[model])
            for model, benchmark in sorted(CANDIDATE_KEYS)
        ],
        _activation_record(*PARTIAL_KEY, layer_by_model[PARTIAL_KEY[0]], partial=True),
    ]
    cache.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    execution.write_text(
        json.dumps(
            {
                "protocol_id": "desired-results-final-test-v1",
                "contract_sha256": "a" * 64,
                "source_publication": {
                    "uri": (
                        "gs://bucket/results/run/org__small/bench_000/"
                        "final-test-v1/execution-v3/publication.json"
                    ),
                    "generation": "1783897693912659",
                    "size": "608",
                    "sha256": "b" * 64,
                },
            },
            sort_keys=True,
        )
    )
    return source, cache, execution


def _connect(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _all_keys(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_migration_and_no_submit_plan_conserve_the_full_matrix_without_fixed_counts(tmp_path):
    source, cache, execution = _create_inputs(tmp_path)
    source_before = _sha256(source)
    output = tmp_path / "inventory-v2.sqlite"
    migration_report = tmp_path / "migration-report.json"

    migrated = inventory.migrate(source, cache, execution, output, migration_report)

    assert _sha256(source) == source_before
    assert migrated["partition"] == {
        "blocked": 16,
        "finalized": 1,
        "activation_complete_candidate": 2,
        "partial": 1,
        "absent": 1480,
        "total": 1500,
    }
    assert sum(
        migrated["partition"][name]
        for name in ("blocked", "finalized", "activation_complete_candidate", "partial", "absent")
    ) == 4 * 375

    with _connect(output) as db:
        finalized = db.execute(
            """
            SELECT t.model_slug,t.benchmark,r.state,r.rerun_locked,
                   r.publication_generation,r.publication_sha256
            FROM targets t JOIN result_state r USING(target_id)
            WHERE r.state='finalized'
            """
        ).fetchall()
        assert [tuple(row) for row in finalized] == [
            ("org__small", "bench_000", "finalized", 1, "1783897693912659", "b" * 64)
        ]
        support = db.execute(
            """
            SELECT s.pair_id,s.stable_id,s.split_name
            FROM target_support s JOIN result_state r USING(target_id)
            WHERE r.state='finalized' ORDER BY s.pair_id
            """
        ).fetchall()
        assert [tuple(row) for row in support] == [
            (0, "stable-final-0", "train"),
            (1, "stable-final-1", "train"),
            (2, "stable-final-2", "train"),
            (3, "stable-final-3", "validation"),
            (4, "stable-final-4", "validation"),
            (5, "stable-final-5", "test"),
        ]
        aggregate_complete = db.execute(
            "SELECT status,eligible,COUNT(*) FROM activation_state WHERE status='complete' GROUP BY status,eligible"
        ).fetchall()
        assert [tuple(row) for row in aggregate_complete] == [("complete", 0, 3)]
        candidate_states = db.execute(
            """
            SELECT t.model_slug,t.benchmark,r.state,a.eligible
            FROM targets t JOIN activation_state a USING(target_id) JOIN result_state r USING(target_id)
            WHERE t.blocked=0 AND a.status='complete' AND r.state='unprepared'
            ORDER BY t.model_slug,t.benchmark
            """
        ).fetchall()
        assert {tuple(row) for row in candidate_states} == {
            (model, benchmark, "unprepared", 0) for model, benchmark in CANDIDATE_KEYS
        }
        assert db.execute("SELECT COUNT(*) FROM method_runs WHERE state='finalized'").fetchone()[0] == 1

    second_output = tmp_path / "inventory-v2-repeat.sqlite"
    repeated = inventory.migrate(
        source,
        cache,
        execution,
        second_output,
        tmp_path / "migration-report-repeat.json",
    )
    assert repeated["inventory_sha256"] == migrated["inventory_sha256"]
    with _connect(second_output) as db:
        assert db.execute("SELECT COUNT(*) FROM result_state WHERE state='finalized'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM method_runs").fetchone()[0] == 1

    plan_dir = tmp_path / "plan"
    plan_report = tmp_path / "plan-report.json"
    planned = inventory.plan(output, plan_dir, plan_report, no_submit=True)
    assert planned["no_submit"] is True
    assert planned["descriptor_count"] == 2
    assert {item["target_id"] for item in planned["descriptors"]} == {
        targets.target_id(PROTOCOL, model, benchmark) for model, benchmark in CANDIDATE_KEYS
    }

    descriptors = [json.loads((plan_dir / item["path"]).read_text()) for item in planned["descriptors"]]
    by_key = {
        (descriptor["target"]["model_slug"], descriptor["target"]["benchmark"]): descriptor
        for descriptor in descriptors
    }
    assert {
        key: (
            descriptor["target"]["layer_count"],
            descriptor["target"]["expected_pairs"],
            len(descriptor["expected_routes"]),
        )
        for key, descriptor in by_key.items()
    } == {
        ("org__small", "bench_001"): (3, 9, 7 * 3),
        ("other__wide", "bench_002"): (5, 6, 7 * 5),
    }
    assert planned["partition"] == migrated["partition"]
    assert planned["descriptor_kind"] == "activation_proof_preflight"
    for descriptor in descriptors:
        assert set(descriptor) == {
            "schema_version", "descriptor_kind", "protocol", "target", "source_evidence",
            "expected_routes", "no_submit", "descriptor_sha256",
        }
        assert descriptor["descriptor_kind"] == "activation_proof_preflight"
        assert descriptor["no_submit"] is True
        unhashed = dict(descriptor)
        digest = unhashed.pop("descriptor_sha256")
        assert digest == targets.canonical_sha256(unhashed)
        assert descriptor["expected_routes"] == [
            {"strategy": strategy, "layer": layer}
            for strategy in targets.STRATEGIES
            for layer in range(1, descriptor["target"]["layer_count"] + 1)
        ]
        all_keys = set(_all_keys(descriptor))
        assert not {"manifest_sha256", "completion_ref", "proof_ref", "uri", "generation", "size"} & all_keys
        assert not {"execution", "support", "activation", "prompt"} & all_keys
        serialized = json.dumps(descriptor, sort_keys=True)
        assert "hf://" not in serialized
        assert "cache://" not in serialized

    with pytest.raises(inventory.InventoryError, match="--no-submit is required"):
        inventory.plan(output, tmp_path / "must-not-run", tmp_path / "bad-report.json", no_submit=False)


def test_execution_import_rejects_cross_target_publication_without_prepared_support(tmp_path):
    source, cache, execution = _create_inputs(tmp_path)
    artifact = json.loads(execution.read_text())
    artifact["source_publication"]["uri"] = (
        "gs://bucket/results/run/other__wide/bench_002/"
        "final-test-v1/execution-v3/publication.json"
    )
    execution.write_text(json.dumps(artifact))

    with pytest.raises(inventory.InventoryError, match="lacks prepared support"):
        inventory.migrate(
            source,
            cache,
            execution,
            tmp_path / "must-not-exist.sqlite",
            tmp_path / "must-not-exist-report.json",
        )
