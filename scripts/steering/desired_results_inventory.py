#!/usr/bin/env python3
"""Build and plan against desired-results inventory schema v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from .desired_results_target import (
        STRATEGIES, canonical_json, canonical_sha256, result_id, target_id,
    )
except ImportError:
    from desired_results_target import (  # type: ignore
        STRATEGIES, canonical_json, canonical_sha256, result_id, target_id,
    )

PROTOCOL_REVISION = 1


class InventoryError(ValueError):
    """Raised when source evidence cannot be migrated without ambiguity."""


DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE targets (
  target_id TEXT PRIMARY KEY,
  result_id TEXT NOT NULL UNIQUE,
  protocol_id TEXT NOT NULL,
  protocol_revision INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  model_slug TEXT NOT NULL,
  benchmark TEXT NOT NULL,
  expected_pairs INTEGER NOT NULL CHECK(expected_pairs > 0),
  layer_count INTEGER,
  result_prefix TEXT NOT NULL UNIQUE,
  blocked INTEGER NOT NULL CHECK(blocked IN (0,1)),
  block_reason_code TEXT,
  block_details TEXT,
  UNIQUE(protocol_id, model_slug, benchmark)
);
CREATE TABLE activation_state (
  target_id TEXT PRIMARY KEY REFERENCES targets(target_id),
  status TEXT NOT NULL CHECK(status IN ('complete','partial','absent')),
  eligible INTEGER NOT NULL CHECK(eligible IN (0,1)),
  n_pairs INTEGER,
  grouped INTEGER,
  layer_count INTEGER,
  strategy_layers_json TEXT NOT NULL,
  cache_sha256 TEXT NOT NULL,
  record_sha256 TEXT,
  CHECK((status='absent' AND n_pairs IS NULL AND grouped IS NULL AND record_sha256 IS NULL) OR status!='absent')
);
CREATE TABLE result_state (
  target_id TEXT PRIMARY KEY REFERENCES targets(target_id),
  state TEXT NOT NULL CHECK(state IN ('unprepared','prepared','calibrated','finalized')),
  rerun_locked INTEGER NOT NULL CHECK(rerun_locked IN (0,1)),
  publication_uri TEXT,
  publication_generation TEXT,
  publication_size TEXT,
  publication_sha256 TEXT,
  execution_sha256 TEXT,
  execution_contract_sha256 TEXT,
  execution_protocol_id TEXT,
  CHECK((state='finalized' AND rerun_locked=1 AND publication_uri IS NOT NULL AND publication_sha256 IS NOT NULL AND execution_sha256 IS NOT NULL) OR state!='finalized')
);
CREATE TABLE target_support (
  target_id TEXT NOT NULL REFERENCES targets(target_id),
  pair_id INTEGER NOT NULL,
  stable_id TEXT NOT NULL,
  split_name TEXT NOT NULL CHECK(split_name IN ('train','validation','test')),
  support_sha256 TEXT NOT NULL,
  PRIMARY KEY(target_id, pair_id),
  UNIQUE(target_id, stable_id)
);
CREATE TABLE method_runs (
  job_key TEXT PRIMARY KEY,
  target_id TEXT NOT NULL REFERENCES targets(target_id),
  method TEXT NOT NULL,
  optimization_run_id TEXT NOT NULL,
  state TEXT NOT NULL,
  staging_prefix TEXT NOT NULL UNIQUE,
  provenance_json TEXT NOT NULL,
  UNIQUE(target_id, method, optimization_run_id)
);
CREATE VIEW counts AS
SELECT partition, COUNT(*) AS count FROM (
  SELECT CASE
    WHEN t.blocked=1 THEN 'blocked'
    WHEN r.state='finalized' THEN 'finalized'
    WHEN a.status='complete' AND r.state='unprepared' THEN 'activation_complete_candidate'
    WHEN a.status='partial' THEN 'partial'
    ELSE 'absent'
  END AS partition
  FROM targets t JOIN activation_state a USING(target_id) JOIN result_state r USING(target_id)
) GROUP BY partition;
"""


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    cache_sha = _file_sha(path)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InventoryError(f"activation cache line {line_number} is invalid JSON: {exc}") from exc
            required = {"tier", "model", "bench", "layers", "n_pairs", "grouped"}
            if not isinstance(record, dict) or set(record) != required:
                raise InventoryError(f"activation cache line {line_number} keys must be exactly {sorted(required)}")
            if record["tier"] != "activations":
                continue
            key = (record["model"], record["bench"])
            if key in records:
                raise InventoryError(f"duplicate activation cache record for {key}")
            if not isinstance(record["layers"], dict) or set(record["layers"]) != set(STRATEGIES):
                raise InventoryError(f"activation cache record {key} lacks the exact seven strategies")
            if any(type(value) is not int or value < 0 for value in record["layers"].values()):
                raise InventoryError(f"activation cache record {key} has invalid layer counts")
            if type(record["n_pairs"]) is not int or record["n_pairs"] < 0 or type(record["grouped"]) is not bool:
                raise InventoryError(f"activation cache record {key} has invalid pair/grouped evidence")
            records[key] = record
    return records, cache_sha




def _model_layer_counts(source: sqlite3.Connection, records: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, int | None]:
    columns = {row[1] for row in source.execute("PRAGMA table_info(models)")}
    rows = source.execute("SELECT * FROM models ORDER BY model_slug").fetchall()
    result: dict[str, int | None] = {}
    for row in rows:
        slug = row["model_slug"]
        if "layer_count" in columns:
            count = row["layer_count"]
        else:
            observed = [
                value for (model, _), record in records.items() if model == slug
                for value in record["layers"].values()
            ]
            count = max(observed) if observed else None
        if count is not None and (type(count) is not int or count <= 0):
            raise InventoryError(f"invalid layer_count for model {slug}")
        result[slug] = count
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json(value) + b"\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _partition_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {name: 0 for name in ("blocked", "finalized", "activation_complete_candidate", "partial", "absent")}
    counts.update({row["partition"]: row["count"] for row in connection.execute("SELECT partition,count FROM counts")})
    counts["total"] = sum(counts.values())
    return counts


def migrate(source: Path, activation_cache: Path, execution_v3: Path, output: Path, report: Path) -> dict[str, Any]:
    """Create a new v2 database; the v1 source is opened read-only and never altered."""
    source, activation_cache, execution_v3, output, report = map(Path, (source, activation_cache, execution_v3, output, report))
    for path in (source, activation_cache, execution_v3):
        if not path.is_file():
            raise InventoryError(f"required input does not exist: {path}")
    if output.resolve() == source.resolve():
        raise InventoryError("output must differ from the immutable source database")
    records, cache_sha = _load_jsonl(activation_cache)
    try:
        execution = json.loads(execution_v3.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"invalid execution-v3 artifact: {exc}") from exc
    publication = execution.get("source_publication")
    if not isinstance(publication, dict) or set(publication) != {"uri", "generation", "size", "sha256"}:
        raise InventoryError("execution-v3 source_publication has an invalid contract")
    if any(not isinstance(publication[key], str) or not publication[key] for key in publication):
        raise InventoryError("execution-v3 publication fields must be non-empty strings")
    for label, digest in (("publication.sha256", publication["sha256"]), ("execution.contract_sha256", execution.get("contract_sha256"))):
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise InventoryError(f"{label} must be a lowercase SHA-256 digest")
    execution_sha = _file_sha(execution_v3)

    source_uri = f"file:{source.resolve()}?mode=ro"
    old = sqlite3.connect(source_uri, uri=True)
    old.row_factory = sqlite3.Row
    try:
        required_tables = {"models", "result_targets", "blocked_targets", "methods"}
        tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required_tables <= tables:
            raise InventoryError(f"source is missing required tables: {sorted(required_tables - tables)}")
        model_rows = old.execute("SELECT * FROM models ORDER BY model_slug").fetchall()
        model_names = {row["model_slug"]: row["model_name"] for row in model_rows}
        layer_counts = _model_layer_counts(old, records)
        target_rows = old.execute("SELECT * FROM result_targets ORDER BY model_slug,benchmark").fetchall()
        blocked_rows = {row["result_id"]: row for row in old.execute("SELECT * FROM blocked_targets")}
        methods = [row[0] for row in old.execute("SELECT method FROM methods ORDER BY method")]
        if not methods:
            raise InventoryError("source methods table is empty")

        prepared: dict[tuple[str, str], sqlite3.Row] = {}
        support_rows: dict[str, list[sqlite3.Row]] = {}
        method_rows: list[sqlite3.Row] = []
        if "prepared_targets" in tables:
            prepared = {(row["model_slug"], row["benchmark"]): row for row in old.execute("SELECT * FROM prepared_targets")}
        if "prepared_target_support" in tables:
            for row in old.execute("SELECT * FROM prepared_target_support ORDER BY target_id,pair_id"):
                support_rows.setdefault(row["target_id"], []).append(row)
        if "prepared_method_runs" in tables:
            method_rows = old.execute("SELECT * FROM prepared_method_runs ORDER BY job_key").fetchall()

        publication_uri = publication["uri"]
        publication_matches = [
            (slug, benchmark) for slug in model_names
            for benchmark in {row["benchmark"] for row in target_rows}
            if f"/{slug}/{benchmark}/" in publication_uri
        ]
        if len(publication_matches) != 1:
            raise InventoryError("execution-v3 publication URI must identify exactly one source target")
        finalized_key = publication_matches[0]
        if finalized_key not in prepared:
            raise InventoryError("finalized execution-v3 target lacks prepared support in source")
        finalized_source_id = prepared[finalized_key]["target_id"]
        finalized_support = support_rows.get(finalized_source_id, [])
        if not finalized_support:
            raise InventoryError("finalized execution-v3 target has no support rows")
        expected_by_key = {(row["model_slug"], row["benchmark"]): row["expected_pairs"] for row in target_rows}
        if len(finalized_support) != expected_by_key[finalized_key]:
            raise InventoryError("finalized support row count does not match expected_pairs")
        pair_ids = [row["pair_id"] for row in finalized_support]
        stable_ids = [row["stable_id"] for row in finalized_support]
        if len(pair_ids) != len(set(pair_ids)) or len(stable_ids) != len(set(stable_ids)):
            raise InventoryError("finalized support identities are not unique")
        if any(row["split_name"] not in {"train", "validation", "test"} for row in finalized_support):
            raise InventoryError("finalized support contains an invalid split")
        support_hash = prepared[finalized_key]["support_hash"]
        if not isinstance(support_hash, str) or len(support_hash) != 64 or any(character not in "0123456789abcdef" for character in support_hash):
            raise InventoryError("finalized support_hash must be a lowercase SHA-256 digest")

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        db = sqlite3.connect(output)
        db.row_factory = sqlite3.Row
        try:
            db.executescript(DDL)
            for row in target_rows:
                protocol = row["protocol_id"]
                slug, benchmark = row["model_slug"], row["benchmark"]
                tid = target_id(protocol, slug, benchmark)
                rid = result_id(protocol, slug, benchmark)
                if row["result_id"] != rid:
                    raise InventoryError(f"source result_id is non-deterministic: {row['result_id']}")
                blocked = blocked_rows.get(rid)
                layer_count = layer_counts.get(slug)
                prefix = f"results/{protocol}/{slug}/{benchmark}"
                db.execute(
                    "INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, rid, protocol, PROTOCOL_REVISION, model_names[slug], slug, benchmark,
                     row["expected_pairs"], layer_count, prefix, int(blocked is not None),
                     blocked["reason_code"] if blocked else None, blocked["details"] if blocked else None),
                )
                record = records.get((slug, benchmark))
                is_finalized = (slug, benchmark) == finalized_key
                zero_layers = {name: 0 for name in STRATEGIES}
                if record is None:
                    status, eligible, n_pairs, grouped, evidence_layers, record_sha = "absent", 0, None, None, zero_layers, None
                else:
                    evidence_layers = {name: record["layers"][name] for name in STRATEGIES}
                    evidence_is_complete = (
                        layer_count is not None and record["n_pairs"] == row["expected_pairs"]
                        and record["grouped"] is False
                        and all(evidence_layers[name] == layer_count for name in STRATEGIES)
                    )
                    status = "complete" if evidence_is_complete else "partial"
                    eligible = 0
                    n_pairs, grouped, record_sha = record["n_pairs"], int(record["grouped"]), canonical_sha256(record)
                db.execute(
                    "INSERT INTO activation_state VALUES (?,?,?,?,?,?,?,?,?)",
                    (tid, status, eligible, n_pairs, grouped, layer_count,
                     canonical_json(evidence_layers).decode("ascii"), cache_sha, record_sha),
                )
                db.execute(
                    "INSERT INTO result_state VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (tid, "finalized" if is_finalized else "unprepared", int(is_finalized),
                     publication["uri"] if is_finalized else None,
                     publication["generation"] if is_finalized else None,
                     publication["size"] if is_finalized else None,
                     publication["sha256"] if is_finalized else None,
                     execution_sha if is_finalized else None,
                     execution.get("contract_sha256") if is_finalized else None,
                     execution.get("protocol_id") if is_finalized else None),
                )
                if is_finalized:
                    support_hash = prepared[finalized_key]["support_hash"]
                    for support in finalized_support:
                        db.execute(
                            "INSERT INTO target_support VALUES (?,?,?,?,?)",
                            (tid, support["pair_id"], support["stable_id"], support["split_name"], support_hash),
                        )
            identities = {
                (row["model_slug"], row["benchmark"]): target_id(row["protocol_id"], row["model_slug"], row["benchmark"])
                for row in target_rows
            }
            source_to_v2 = {
                row["target_id"]: identities[(row["model_slug"], row["benchmark"])]
                for row in prepared.values()
            }
            finalized_target_id = identities[finalized_key]
            for row in method_rows:
                if row["target_id"] not in source_to_v2:
                    continue
                tid = source_to_v2[row["target_id"]]
                if tid != finalized_target_id:
                    continue
                provenance = {key: row[key] for key in row.keys() if key not in {"job_key", "target_id", "method", "optimization_run_id", "staging_prefix"}}
                job_key = f"{tid}:{row['method']}:{row['optimization_run_id']}"
                db.execute("INSERT INTO method_runs VALUES (?,?,?,?,?,?,?)", (
                    job_key, tid, row["method"], row["optimization_run_id"], "finalized",
                    row["staging_prefix"], canonical_json(provenance).decode("ascii"),
                ))
            db.commit()
            partition = _partition_counts(db)
            logical_rows = {
                table: [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
                for table in ("targets", "activation_state", "result_state", "target_support", "method_runs")
            }
            inventory_sha = canonical_sha256(logical_rows)
        finally:
            db.close()
    finally:
        old.close()

    report_value = {
        "schema_version": 2,
        "inventory_sha256": inventory_sha,
        "activation_cache_sha256": cache_sha,
        "execution_v3_sha256": execution_sha,
        "partition": partition,
        "output": str(output),
    }
    _atomic_json(report, report_value)
    return report_value


def _finalize_preflight_descriptor(payload: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = dict(payload)
    if "descriptor_sha256" in descriptor:
        raise InventoryError("preflight descriptor payload already contains descriptor_sha256")
    descriptor["descriptor_sha256"] = canonical_sha256(descriptor)
    return descriptor


def plan(inventory: Path, output_dir: Path, report: Path, no_submit: bool) -> dict[str, Any]:
    """Write proof-collection preflight descriptors; never create or submit execution manifests."""
    inventory, output_dir, report = map(Path, (inventory, output_dir, report))
    if no_submit is not True:
        raise InventoryError("plan is non-submitting; --no-submit is required")
    if not inventory.is_file():
        raise InventoryError(f"inventory does not exist: {inventory}")
    db = sqlite3.connect(f"file:{inventory.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        logical_rows = {
            table: [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("targets", "activation_state", "result_state", "target_support", "method_runs")
        }
        inventory_sha = canonical_sha256(logical_rows)
        cache_hashes = {row[0] for row in db.execute("SELECT DISTINCT cache_sha256 FROM activation_state")}
        if len(cache_hashes) != 1:
            raise InventoryError("inventory has inconsistent activation cache identities")
        cache_sha = next(iter(cache_hashes))
        rows = db.execute("""
          SELECT t.*, a.status activation_status, a.eligible activation_eligible,
                 a.n_pairs, a.grouped, a.layer_count activation_layer_count,
                 a.strategy_layers_json, a.record_sha256,
                 r.state result_status, r.rerun_locked
          FROM targets t JOIN activation_state a USING(target_id) JOIN result_state r USING(target_id)
          WHERE t.blocked=0 AND a.status='complete' AND r.state='unprepared' AND r.rerun_locked=0
          ORDER BY t.model_slug,t.benchmark
        """).fetchall()
        descriptors = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row["activation_eligible"] != 0:
                raise InventoryError("aggregate activation coverage cannot be execution eligible before route proof collection")
            layer_count = row["activation_layer_count"]
            if type(layer_count) is not int or layer_count <= 0:
                raise InventoryError(f"candidate {row['target_id']} lacks model layer_count")
            expected_routes = [
                {"strategy": strategy, "layer": layer}
                for strategy in STRATEGIES for layer in range(1, layer_count + 1)
            ]
            payload = {
                "schema_version": 2,
                "descriptor_kind": "activation_proof_preflight",
                "protocol": {"id": row["protocol_id"], "revision": row["protocol_revision"]},
                "target": {
                    "target_id": row["target_id"], "result_id": row["result_id"],
                    "model_name": row["model_name"], "model_slug": row["model_slug"],
                    "benchmark": row["benchmark"], "expected_pairs": row["expected_pairs"],
                    "layer_count": layer_count,
                },
                "source_evidence": {
                    "inventory_sha256": inventory_sha,
                    "activation_cache_sha256": cache_sha,
                    "activation_record_sha256": row["record_sha256"],
                    "observed_n_pairs": row["n_pairs"],
                    "observed_grouped": bool(row["grouped"]),
                    "observed_strategy_layers": json.loads(row["strategy_layers_json"]),
                },
                "expected_routes": expected_routes,
                "no_submit": True,
            }
            descriptor = _finalize_preflight_descriptor(payload)
            filename = f"{descriptor['descriptor_sha256']}.json"
            _atomic_json(output_dir / filename, descriptor)
            descriptors.append({
                "target_id": row["target_id"], "descriptor_sha256": descriptor["descriptor_sha256"],
                "path": filename,
            })
        partition = _partition_counts(db)
    finally:
        db.close()
    result = {
        "schema_version": 2, "inventory_sha256": inventory_sha, "no_submit": True,
        "descriptor_kind": "activation_proof_preflight",
        "descriptor_count": len(descriptors), "descriptors": descriptors, "partition": partition,
    }
    _atomic_json(report, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migration = commands.add_parser("migrate")
    migration.add_argument("--source", type=Path, required=True)
    migration.add_argument("--activation-cache", type=Path, required=True)
    migration.add_argument("--execution-v3", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    migration.add_argument("--report", type=Path, required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--inventory", type=Path, required=True)
    planning.add_argument("--output-dir", type=Path, required=True)
    planning.add_argument("--report", type=Path, required=True)
    planning.add_argument("--no-submit", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "migrate":
        value = migrate(args.source, args.activation_cache, args.execution_v3, args.output, args.report)
    else:
        value = plan(args.inventory, args.output_dir, args.report, args.no_submit)
    print(canonical_json(value).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
