#!/usr/bin/env python3
"""Strict, content-addressed target descriptor contract for desired results."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

SCHEMA_VERSION = 2
ACTIVATION_STATES = frozenset({"complete", "partial", "absent"})
RESULT_STATES = frozenset({"unprepared", "prepared", "calibrated", "finalized"})
SUPPORT_STATES = frozenset({"missing", "prepared"})
TOP_KEYS = frozenset({
    "schema_version", "protocol", "target", "revisions", "activation",
    "support", "evaluation", "calibration", "execution", "manifest_sha256",
})
STRATEGIES = (
    "chat_first", "chat_last", "chat_max_norm", "chat_mean",
    "chat_weighted", "mc_balanced", "role_play",
)
METHODS = ("caa", "ostrze", "mlp", "tecza", "tetno", "grom", "nurt", "wicher")


class ContractError(ValueError):
    """Raised when a target descriptor violates schema v2."""


def canonical_json(value: Any) -> bytes:
    """Return the sole canonical JSON representation used by this contract."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical-JSON encodable: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def target_id(protocol: str, model_slug: str, benchmark: str) -> str:
    return f"target-v2:{_identity_digest(protocol, model_slug, benchmark)}"


def result_id(protocol: str, model_slug: str, benchmark: str) -> str:
    return f"result-v2:{_identity_digest(protocol, model_slug, benchmark)}"


def _identity_digest(protocol: str, model_slug: str, benchmark: str) -> str:
    _identity_parts(protocol, model_slug, benchmark)
    return canonical_sha256({
        "protocol": protocol, "model_slug": model_slug, "benchmark": benchmark,
    })


def _identity_parts(protocol: str, model_slug: str, benchmark: str) -> None:
    for label, part in (("protocol", protocol), ("model_slug", model_slug)):
        if (not isinstance(part, str) or not part or part in {".", ".."}
                or any(character in part for character in (":", "/", "\\", "\x00"))):
            raise ContractError(f"{label} must be a non-empty path-safe identity component")
    if (not isinstance(benchmark, str) or not benchmark or benchmark.startswith("/")
            or any(character in benchmark for character in (":", "\\", "\x00"))):
        raise ContractError("benchmark must be a safe non-absolute category path")
    if any(part in {"", ".", ".."} for part in benchmark.split("/")):
        raise ContractError("benchmark must be a safe non-absolute category path")


def _exact(value: Any, keys: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        actual = set(value) if isinstance(value, Mapping) else type(value).__name__
        raise ContractError(f"{label} keys must be exactly {sorted(keys)}; got {actual}")
    return value


def _string(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ContractError(f"{label} must be {'a string' if empty else 'a non-empty string'}")
    return value


def _sha(value: Any, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{label} must be a lowercase SHA-256 hex digest")


def _revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{label} must be a lowercase 40-character Git revision")
    return value


def _artifact_ref(value: Any, label: str) -> Mapping[str, Any]:
    ref = _exact(value, {"uri", "generation", "size", "sha256"}, label)
    _string(ref["uri"], f"{label}.uri")
    _string(ref["generation"], f"{label}.generation")
    size = _string(ref["size"], f"{label}.size")
    if not size.isascii() or not size.isdecimal() or size.startswith("0"):
        raise ContractError(f"{label}.size must be a canonical positive decimal string")
    _sha(ref["sha256"], f"{label}.sha256")
    return ref


def _positive(value: Any, label: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _strings(value: Any, label: str, *, unique: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"{label} must be a list of non-empty strings")
    if unique and len(value) != len(set(value)):
        raise ContractError(f"{label} contains duplicates")
    return value


def finalize_target_manifest(payload_without_hash: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload_without_hash, Mapping):
        raise ContractError("manifest payload must be an object")
    payload = dict(payload_without_hash)
    if "manifest_sha256" in payload:
        raise ContractError("finalize payload must not already contain manifest_sha256")
    payload["manifest_sha256"] = canonical_sha256(payload)
    validate_target_manifest(payload)
    return payload


def validate_target_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on every schema, identity, state, count, and hash invariant."""
    root = _exact(manifest, TOP_KEYS, "manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError("schema_version must be 2")

    protocol = _exact(root["protocol"], {"id", "revision"}, "protocol")
    protocol_id = _string(protocol["id"], "protocol.id")
    _positive(protocol["revision"], "protocol.revision")

    target = _exact(root["target"], {
        "target_id", "result_id", "model_name", "model_slug", "benchmark",
        "expected_pairs", "result_prefix",
    }, "target")
    model_slug = _string(target["model_slug"], "target.model_slug")
    benchmark = _string(target["benchmark"], "target.benchmark")
    _identity_parts(protocol_id, model_slug, benchmark)
    _string(target["model_name"], "target.model_name")
    expected_pairs = _positive(target["expected_pairs"], "target.expected_pairs")
    expected_id = target_id(protocol_id, model_slug, benchmark)
    if target["target_id"] != expected_id or target["result_id"] != result_id(protocol_id, model_slug, benchmark):
        raise ContractError("target_id/result_id do not match protocol, model_slug, and benchmark")
    expected_prefix = f"results/{protocol_id}/{model_slug}/{benchmark}"
    if target["result_prefix"] != expected_prefix:
        raise ContractError(f"result_prefix must be {expected_prefix!r}")

    revisions = _exact(root["revisions"], {
        "inventory_sha256", "model_revision", "tokenizer_revision", "activation_revision",
    }, "revisions")
    _sha(revisions["inventory_sha256"], "revisions.inventory_sha256")
    for name in ("model_revision", "tokenizer_revision", "activation_revision"):
        _revision(revisions[name], f"revisions.{name}")

    activation = _exact(root["activation"], {
        "status", "eligible", "layer_count", "n_pairs", "grouped", "strategies", "routes", "proof",
    }, "activation")
    status = activation["status"]
    if status not in ACTIVATION_STATES or type(activation["eligible"]) is not bool:
        raise ContractError("activation status/eligible is invalid")
    layer_count = activation["layer_count"]
    n_pairs = activation["n_pairs"]
    if layer_count is not None:
        _positive(layer_count, "activation.layer_count")
    if n_pairs is not None:
        _positive(n_pairs, "activation.n_pairs", zero=True)
    if activation["grouped"] is not None and type(activation["grouped"]) is not bool:
        raise ContractError("activation.grouped must be boolean or null")
    strategies = _exact(activation["strategies"], set(STRATEGIES), "activation.strategies")
    for strategy, count in strategies.items():
        _positive(count, f"activation.strategies.{strategy}", zero=True)
    proof = _exact(activation["proof"], {"cache_sha256", "record_sha256"}, "activation.proof")
    _sha(proof["cache_sha256"], "activation.proof.cache_sha256")
    _sha(proof["record_sha256"], "activation.proof.record_sha256", optional=True)
    routes = activation["routes"]
    if not isinstance(routes, list):
        raise ContractError("activation.routes must be a list")
    route_keys: set[tuple[str, int]] = set()
    for index, route_value in enumerate(routes):
        route = _exact(route_value, {"strategy", "layer", "completion_ref", "proof_ref"}, f"activation.routes[{index}]")
        strategy = route["strategy"]
        if strategy not in STRATEGIES:
            raise ContractError(f"activation.routes[{index}].strategy is invalid")
        layer = _positive(route["layer"], f"activation.routes[{index}].layer")
        key = (strategy, layer)
        if key in route_keys:
            raise ContractError("activation.routes contains a duplicate strategy/layer")
        route_keys.add(key)
        for ref_name in ("completion_ref", "proof_ref"):
            ref = _exact(route[ref_name], {"uri", "generation", "size", "sha256"}, f"activation.routes[{index}].{ref_name}")
            for field in ("uri", "generation", "size"):
                _string(ref[field], f"activation.routes[{index}].{ref_name}.{field}")
            _sha(ref["sha256"], f"activation.routes[{index}].{ref_name}.sha256")
            identity_path = f"/{model_slug}/{benchmark}/" in ref["uri"]
            content_addressed_path = (
                ref["uri"].startswith("gs://")
                and ref["uri"].endswith(f"/artifacts/{ref['sha256']}.json")
            )
            if not identity_path and not content_addressed_path:
                raise ContractError(
                    "activation route reference is neither target-scoped nor canonically content-addressed"
                )
    expected_routes = set() if layer_count is None else {(strategy, layer) for strategy in STRATEGIES for layer in range(1, layer_count + 1)}
    evidence_complete = (
        status == "complete" and layer_count is not None and n_pairs == expected_pairs
        and activation["grouped"] is False
        and all(strategies[name] == layer_count for name in STRATEGIES)
        and route_keys == expected_routes
    )
    if status == "complete" and not evidence_complete:
        raise ContractError("complete activation lacks the exact pair/strategy/layer proof matrix")
    if status == "partial" and (proof["record_sha256"] is None or not route_keys <= expected_routes):
        raise ContractError("partial activation must carry valid incomplete record evidence")
    if status == "absent" and any((n_pairs is not None, activation["grouped"] is not None, proof["record_sha256"] is not None, any(strategies.values()), routes)):
        raise ContractError("absent activation cannot carry record evidence")

    support = _exact(root["support"], {
        "state", "proof_sha256", "pair_count", "split_counts", "splits", "pair_texts_ref",
    }, "support")
    if support["state"] not in SUPPORT_STATES:
        raise ContractError("support.state is invalid")
    _sha(support["proof_sha256"], "support.proof_sha256", optional=True)
    pair_texts_ref = support["pair_texts_ref"]
    if pair_texts_ref is not None:
        _artifact_ref(pair_texts_ref, "support.pair_texts_ref")
    pair_count = _positive(support["pair_count"], "support.pair_count", zero=True)
    split_names = {"train", "validation", "test"}
    split_counts = _exact(support["split_counts"], split_names, "support.split_counts")
    splits = _exact(support["splits"], split_names, "support.splits")
    pair_ids: set[int] = set()
    stable_ids: set[str] = set()
    for name in split_names:
        rows = splits[name]
        if not isinstance(rows, list):
            raise ContractError(f"support.splits.{name} must be a list")
        if split_counts[name] != len(rows):
            raise ContractError(f"support split count mismatch for {name}")
        for index, row_value in enumerate(rows):
            row = _exact(row_value, {"pair_id", "stable_id"}, f"support.splits.{name}[{index}]")
            pair_id = _positive(row["pair_id"], f"support.splits.{name}[{index}].pair_id", zero=True)
            stable_id = _string(row["stable_id"], f"support.splits.{name}[{index}].stable_id")
            if pair_id in pair_ids or stable_id in stable_ids:
                raise ContractError("support pair_id and stable_id must each be globally unique")
            pair_ids.add(pair_id)
            stable_ids.add(stable_id)
    if pair_count != sum(split_counts.values()):
        raise ContractError("support.pair_count must equal split_counts total")
    if support["state"] == "missing" and (
        pair_count != 0 or support["proof_sha256"] is not None or pair_texts_ref is not None
    ):
        raise ContractError("missing support cannot carry pairs, proof, or pair texts")
    if support["state"] == "prepared" and (
        pair_count != expected_pairs or support["proof_sha256"] is None or pair_texts_ref is None
    ):
        raise ContractError("prepared support must prove exactly expected_pairs and bind pair texts")

    evaluation = _exact(root["evaluation"], {"required_outputs", "split"}, "evaluation")
    required_outputs = _strings(evaluation["required_outputs"], "evaluation.required_outputs")
    if not required_outputs:
        raise ContractError("evaluation.required_outputs cannot be empty")
    if evaluation["split"] not in split_names:
        raise ContractError("evaluation.split is invalid")

    calibration = _exact(root["calibration"], {"methods", "strategies", "layer_count", "expected_pairs"}, "calibration")
    if _strings(calibration["methods"], "calibration.methods") != list(METHODS):
        raise ContractError("calibration.methods must be the exact ordered eight-method matrix")
    if calibration["strategies"] != list(STRATEGIES):
        raise ContractError("calibration.strategies must be the required seven-strategy route")
    if calibration["layer_count"] is not None:
        _positive(calibration["layer_count"], "calibration.layer_count")
    if calibration["layer_count"] != layer_count or calibration["expected_pairs"] != expected_pairs:
        raise ContractError("calibration counts must match target activation evidence")

    execution = _exact(root["execution"], {"state", "blocked", "rerun_locked", "publication", "provenance"}, "execution")
    result_state = execution["state"]
    if result_state not in RESULT_STATES or type(execution["blocked"]) is not bool or type(execution["rerun_locked"]) is not bool:
        raise ContractError("execution state/flags are invalid")
    publication = execution["publication"]
    if publication is not None:
        publication = _exact(publication, {"uri", "generation", "size", "sha256"}, "execution.publication")
        for key in ("uri", "generation", "size"):
            _string(publication[key], f"execution.publication.{key}")
        _sha(publication["sha256"], "execution.publication.sha256")
    provenance = _exact(execution["provenance"], {"execution_sha256", "contract_sha256"}, "execution.provenance")
    _sha(provenance["execution_sha256"], "execution.provenance.execution_sha256", optional=True)
    _sha(provenance["contract_sha256"], "execution.provenance.contract_sha256", optional=True)
    if result_state == "finalized":
        if publication is None or not execution["rerun_locked"] or any(value is None for value in provenance.values()):
            raise ContractError("finalized execution requires immutable publication, provenance, and rerun lock")
        publication_identity = f"/{model_slug}/{benchmark}/"
        if publication_identity not in publication["uri"]:
            raise ContractError("publication URI does not match target model/benchmark identity")
    elif publication is not None or any(value is not None for value in provenance.values()):
        raise ContractError("non-finalized execution cannot claim publication provenance")
    expected_eligible = evidence_complete and result_state == "unprepared" and not execution["blocked"] and not execution["rerun_locked"]
    if activation["eligible"] != expected_eligible:
        raise ContractError("activation eligibility does not match activation proof and execution state")
    if result_state in {"prepared", "calibrated", "finalized"} and support["state"] != "prepared":
        raise ContractError("prepared-or-later result requires prepared support")

    digest = root["manifest_sha256"]
    _sha(digest, "manifest_sha256")
    unhashed = dict(root)
    del unhashed["manifest_sha256"]
    if digest != canonical_sha256(unhashed):
        raise ContractError("manifest_sha256 does not match canonical manifest payload")
