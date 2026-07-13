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
from urllib.parse import urlsplit

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
  model_revision TEXT NOT NULL,
  tokenizer_revision TEXT NOT NULL,
  benchmark TEXT NOT NULL,
  expected_pairs INTEGER NOT NULL CHECK(expected_pairs >= 0),
  layer_count INTEGER,
  result_prefix TEXT NOT NULL UNIQUE,
  blocked INTEGER NOT NULL CHECK(blocked IN (0,1)),
  block_reason_code TEXT,
  block_details TEXT,
  CHECK(expected_pairs > 0 OR blocked=1),
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
  activation_repo_id TEXT NOT NULL,
  activation_repo_type TEXT NOT NULL,
  activation_revision TEXT NOT NULL,
  CHECK((status='absent' AND n_pairs IS NULL) OR (status!='absent' AND n_pairs IS NOT NULL))
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
            if not isinstance(record["model"], str) or not record["model"] or not isinstance(record["bench"], str) or not record["bench"]:
                raise InventoryError(f"activation cache line {line_number} has invalid model/benchmark identity")
            key = (record["model"], record["bench"])
            if key in records:
                raise InventoryError(f"duplicate activation cache record for {key}")
            layers = record["layers"]
            if not isinstance(layers, dict):
                raise InventoryError(f"activation cache record {key} layers must be an object")
            unknown = sorted(set(layers) - set(STRATEGIES))
            if unknown:
                raise InventoryError(f"activation cache record {key} has unknown strategies: {unknown}")
            n_pairs, grouped = record["n_pairs"], record["grouped"]
            if n_pairs is None:
                if layers or grouped is not False:
                    raise InventoryError(f"activation cache record {key} has invalid pair/grouped evidence")
            elif not layers or type(n_pairs) is not int or n_pairs < 0 or type(grouped) is not bool:
                raise InventoryError(f"activation cache record {key} has invalid pair/grouped evidence")
            if any(type(value) is not int or value < 0 for value in layers.values()):
                raise InventoryError(f"activation cache record {key} has invalid layer counts")
            records[key] = record
    return records, cache_sha




def _model_layer_counts(source: sqlite3.Connection) -> dict[str, int | None]:
    columns = {row[1] for row in source.execute("PRAGMA table_info(models)")}
    rows = source.execute("SELECT * FROM models ORDER BY model_slug").fetchall()
    result: dict[str, int | None] = {}
    for row in rows:
        slug = row["model_slug"]
        count = row["layer_count"] if "layer_count" in columns else None
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


def _commit_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise InventoryError(f"{label} must be an exact lowercase 40-character commit SHA")
    return value


def _revision_map(value: Any, model_names: set[str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise InventoryError(f"{label} must be a mapping keyed by exact model_name")
    actual = set(value)
    missing, extra = sorted(model_names - actual), sorted(actual - model_names)
    if missing or extra:
        raise InventoryError(f"{label} keys must match source model names; missing={missing}, extra={extra}")
    return {name: _commit_sha(value[name], f"{label}[{name!r}]") for name in sorted(model_names)}


def _layer_count_map(value: Any, model_slugs: set[str], label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise InventoryError(f"{label} must be a mapping keyed by exact model_slug")
    actual = set(value)
    missing, extra = sorted(model_slugs - actual), sorted(actual - model_slugs)
    if missing or extra:
        raise InventoryError(
            f"{label} keys must match source model slugs; missing={missing}, extra={extra}"
        )
    result: dict[str, int] = {}
    for slug in sorted(model_slugs):
        count = value[slug]
        if type(count) is not int or count <= 0:
            raise InventoryError(f"{label}[{slug!r}] must be a positive integer")
        result[slug] = count
    return result


def _positive_decimal(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InventoryError(f"execution-v3 publication fields must be non-empty strings ({label})")
    if not value.isascii() or not value.isdecimal() or value.startswith("0"):
        raise InventoryError(f"{label} must be a canonical positive decimal string")
    return value


def _publication_target(uri: str, target_keys: set[tuple[str, str]]) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if f"{parsed.scheme}://{parsed.netloc}{parsed.path}" != uri:
        raise InventoryError("execution-v3 publication URI must be immutable without query or fragment")
    if parsed.scheme != "gs" or not parsed.netloc:
        raise InventoryError("execution-v3 publication URI must be a gs:// object URI")
    encoded_segments = parsed.path.removeprefix("/").split("/")
    if any(not segment for segment in encoded_segments):
        raise InventoryError("execution-v3 publication URI has empty path segments")
    suffix = ["final-test-v1", "execution-v3", "publication.json"]
    if encoded_segments[-len(suffix):] != suffix:
        raise InventoryError("execution-v3 publication URI has an invalid publication suffix")
    target_segments = encoded_segments[:-len(suffix)]
    matches = []
    for slug, benchmark in sorted(target_keys):
        identity = [slug, *benchmark.split("/")]
        if len(target_segments) >= len(identity) and target_segments[-len(identity):] == identity:
            matches.append((slug, benchmark))
    if len(matches) != 1:
        raise InventoryError("execution-v3 publication URI must identify exactly one complete source target path")
    return matches[0]


def migrate(
    source: Path,
    activation_cache: Path,
    execution_v3: Path,
    output: Path,
    report: Path,
    activation_repo_id: str,
    activation_repo_type: str,
    activation_revision: str,
    model_revisions: Mapping[str, str],
    tokenizer_revisions: Mapping[str, str],
    model_layer_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Create a new v2 database; the v1 source is opened read-only and never altered."""
    source, activation_cache, execution_v3, output, report = map(
        Path, (source, activation_cache, execution_v3, output, report)
    )
    for path in (source, activation_cache, execution_v3):
        if not path.is_file():
            raise InventoryError(f"required input does not exist: {path}")
    if output.resolve() == source.resolve():
        raise InventoryError("output must differ from the immutable source database")
    if not isinstance(activation_repo_id, str) or not activation_repo_id.strip():
        raise InventoryError("activation_repo_id must be a non-empty string")
    if activation_repo_type not in {"dataset", "model", "space"}:
        raise InventoryError("activation_repo_type must be one of dataset, model, or space")
    activation_revision = _commit_sha(activation_revision, "activation_revision")
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
    _positive_decimal(publication["generation"], "publication.generation")
    _positive_decimal(publication["size"], "publication.size")
    for label, digest in (
        ("publication.sha256", publication["sha256"]),
        ("execution.contract_sha256", execution.get("contract_sha256")),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise InventoryError(f"{label} must be a lowercase SHA-256 digest")
    execution_sha = _file_sha(execution_v3)

    source_uri = f"file:{source.resolve()}?mode=ro"
    old = sqlite3.connect(source_uri, uri=True)
    old.row_factory = sqlite3.Row
    temporary_output: str | None = None
    try:
        required_tables = {"models", "result_targets", "blocked_targets", "methods"}
        tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required_tables <= tables:
            raise InventoryError(f"source is missing required tables: {sorted(required_tables - tables)}")
        model_rows = old.execute("SELECT * FROM models ORDER BY model_slug").fetchall()
        model_names = {row["model_slug"]: row["model_name"] for row in model_rows}
        exact_model_revisions = _revision_map(model_revisions, set(model_names.values()), "model_revisions")
        exact_tokenizer_revisions = _revision_map(tokenizer_revisions, set(model_names.values()), "tokenizer_revisions")
        exact_layer_counts = _layer_count_map(
            model_layer_counts, set(model_names), "model_layer_counts",
        )
        source_layer_counts = _model_layer_counts(old)
        for slug, source_count in source_layer_counts.items():
            if source_count is not None and source_count != exact_layer_counts[slug]:
                raise InventoryError(f"source layer_count differs from model_layer_counts for {slug}")
        layer_counts = dict(exact_layer_counts)
        target_rows = old.execute("SELECT * FROM result_targets ORDER BY model_slug,benchmark").fetchall()
        blocked_rows = {row["result_id"]: row for row in old.execute("SELECT * FROM blocked_targets")}
        methods = [row[0] for row in old.execute("SELECT method FROM methods ORDER BY method")]
        if not methods:
            raise InventoryError("source methods table is empty")

        target_keys: set[tuple[str, str]] = set()
        for row in target_rows:
            protocol, slug, benchmark = row["protocol_id"], row["model_slug"], row["benchmark"]
            if slug not in model_names:
                raise InventoryError(f"source target references unknown model_slug {slug!r}")
            legacy_result_id = f"{protocol}:{slug}:{benchmark}"
            if row["result_id"] != legacy_result_id:
                raise InventoryError(f"source result_id is not the exact legacy v1 identity: {row['result_id']!r}")
            key = (slug, benchmark)
            if key in target_keys:
                raise InventoryError(f"duplicate source target {key}")
            target_keys.add(key)
            expected_pairs = row["expected_pairs"]
            if type(expected_pairs) is not int or expected_pairs < 0:
                raise InventoryError(f"source target {key} has invalid expected_pairs")
            if expected_pairs == 0 and legacy_result_id not in blocked_rows:
                raise InventoryError(f"unblocked source target {key} must have expected_pairs > 0")

        prepared: dict[tuple[str, str], sqlite3.Row] = {}
        support_rows: dict[str, list[sqlite3.Row]] = {}
        method_rows: list[sqlite3.Row] = []
        if "prepared_targets" in tables:
            for row in old.execute("SELECT * FROM prepared_targets ORDER BY model_slug,benchmark"):
                key = (row["model_slug"], row["benchmark"])
                if key in prepared:
                    raise InventoryError(f"duplicate prepared target {key}")
                prepared[key] = row
        if "prepared_target_support" in tables:
            for row in old.execute("SELECT * FROM prepared_target_support ORDER BY target_id,pair_id"):
                support_rows.setdefault(row["target_id"], []).append(row)
        if "prepared_method_runs" in tables:
            method_rows = old.execute("SELECT * FROM prepared_method_runs ORDER BY job_key").fetchall()

        finalized_key = _publication_target(publication["uri"], target_keys)
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
        descriptor, temporary_output = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        os.close(descriptor)
        os.unlink(temporary_output)
        db = sqlite3.connect(temporary_output)
        db.row_factory = sqlite3.Row
        try:
            db.executescript(DDL)
            for row in target_rows:
                protocol = row["protocol_id"]
                slug, benchmark = row["model_slug"], row["benchmark"]
                tid = target_id(protocol, slug, benchmark)
                rid = result_id(protocol, slug, benchmark)
                legacy_rid = f"{protocol}:{slug}:{benchmark}"
                blocked = blocked_rows.get(legacy_rid)
                layer_count = layer_counts.get(slug)
                model_name = model_names[slug]
                prefix = f"results/{protocol}/{slug}/{benchmark}"
                db.execute(
                    "INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, rid, protocol, PROTOCOL_REVISION, model_name, slug,
                     exact_model_revisions[model_name], exact_tokenizer_revisions[model_name],
                     benchmark, row["expected_pairs"], layer_count, prefix, int(blocked is not None),
                     blocked["reason_code"] if blocked else None, blocked["details"] if blocked else None),
                )
                record = records.get((slug, benchmark))
                is_finalized = (slug, benchmark) == finalized_key
                zero_layers = {name: 0 for name in STRATEGIES}
                if record is None:
                    status, eligible, n_pairs, grouped = "absent", 0, None, None
                    evidence_layers, record_sha = zero_layers, None
                elif record["n_pairs"] is None:
                    status, eligible, n_pairs, grouped = "absent", 0, None, int(record["grouped"])
                    evidence_layers, record_sha = zero_layers, canonical_sha256(record)
                else:
                    evidence_layers = {name: record["layers"].get(name, 0) for name in STRATEGIES}
                    evidence_is_complete = (
                        layer_count is not None and record["n_pairs"] == row["expected_pairs"]
                        and record["grouped"] is False
                        and all(evidence_layers[name] == layer_count for name in STRATEGIES)
                    )
                    status = "complete" if evidence_is_complete else "partial"
                    eligible = 0
                    n_pairs, grouped, record_sha = record["n_pairs"], int(record["grouped"]), canonical_sha256(record)
                db.execute(
                    "INSERT INTO activation_state VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, status, eligible, n_pairs, grouped, layer_count,
                     canonical_json(evidence_layers).decode("ascii"), cache_sha, record_sha,
                     activation_repo_id, activation_repo_type, activation_revision),
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
                for row in prepared.values() if (row["model_slug"], row["benchmark"]) in identities
            }
            finalized_target_id = identities[finalized_key]
            for row in method_rows:
                tid = source_to_v2.get(row["target_id"])
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
                table: [dict(item) for item in db.execute(f"SELECT * FROM {table} ORDER BY 1")]
                for table in ("targets", "activation_state", "result_state", "target_support", "method_runs")
            }
            inventory_sha = canonical_sha256(logical_rows)
        finally:
            db.close()

        report_value = {
            "schema_version": 2,
            "inventory_sha256": inventory_sha,
            "activation_cache_sha256": cache_sha,
            "execution_v3_sha256": execution_sha,
            "activation_source": {
                "repo_id": activation_repo_id,
                "repo_type": activation_repo_type,
                "revision": activation_revision,
            },
            "model_revisions": exact_model_revisions,
            "tokenizer_revisions": exact_tokenizer_revisions,
            "model_layer_counts": exact_layer_counts,
            "partition": partition,
            "output": str(output),
        }
        os.replace(temporary_output, output)
        temporary_output = None
        _atomic_json(report, report_value)
        return report_value
    finally:
        old.close()
        if temporary_output is not None:
            try:
                os.unlink(temporary_output)
            except FileNotFoundError:
                pass


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
                 a.activation_repo_id, a.activation_repo_type, a.activation_revision,
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
                    "activation_repo_id": row["activation_repo_id"],
                    "activation_repo_type": row["activation_repo_type"],
                    "activation_revision": row["activation_revision"],
                    "model_revision": row["model_revision"],
                    "tokenizer_revision": row["tokenizer_revision"],
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


def _load_revision_argument(value: str, label: str) -> Mapping[str, str]:
    candidate = Path(value)
    try:
        serialized = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
        parsed = json.loads(serialized)
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"{label} must be a JSON object or a path to one: {exc}") from exc
    if not isinstance(parsed, dict):
        raise InventoryError(f"{label} must be a JSON object keyed by exact model_name")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migration = commands.add_parser("migrate")
    migration.add_argument("--source", type=Path, required=True)
    migration.add_argument("--activation-cache", type=Path, required=True)
    migration.add_argument("--execution-v3", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    migration.add_argument("--report", type=Path, required=True)
    migration.add_argument("--activation-repo-id", required=True)
    migration.add_argument("--activation-repo-type", required=True)
    migration.add_argument("--activation-revision", required=True)
    migration.add_argument("--model-layer-counts", required=True)
    migration.add_argument("--model-revisions", required=True)
    migration.add_argument("--tokenizer-revisions", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--inventory", type=Path, required=True)
    planning.add_argument("--output-dir", type=Path, required=True)
    planning.add_argument("--report", type=Path, required=True)
    planning.add_argument("--no-submit", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "migrate":
        value = migrate(
            args.source, args.activation_cache, args.execution_v3, args.output, args.report,
            args.activation_repo_id, args.activation_repo_type, args.activation_revision,
            _load_revision_argument(args.model_revisions, "model_revisions"),
            _load_revision_argument(args.tokenizer_revisions, "tokenizer_revisions"),
            _load_revision_argument(args.model_layer_counts, "model_layer_counts"),
        )
    else:
        value = plan(args.inventory, args.output_dir, args.report, args.no_submit)
    print(canonical_json(value).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
