#!/usr/bin/env python3
"""Create-only production publisher for a desired-results policy bundle.

The CLI validates the complete plan before importing the GCS client and writing
generation-pinned objects.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

try:
    from .desired_results_execution_contract import (
        ContractError,
        canonical_json,
        canonical_sha256,
        validate_artifact_binding,
        validate_artifact_ref,
    )
    from .desired_results_policy import validate_policy_bundle
except (ImportError, ModuleNotFoundError):
    _POLICY_PATH = Path(__file__).with_name("desired_results_policy.py")
    _POLICY_SPEC = importlib.util.spec_from_file_location("desired_results_policy_v3", _POLICY_PATH)
    if _POLICY_SPEC is None or _POLICY_SPEC.loader is None:
        raise ImportError(f"cannot load policy module from {_POLICY_PATH}")
    _policy = importlib.util.module_from_spec(_POLICY_SPEC)
    _POLICY_SPEC.loader.exec_module(_policy)
    ContractError = _policy.ContractError
    canonical_json = _policy.canonical_json
    canonical_sha256 = _policy.canonical_sha256
    validate_artifact_binding = _policy.validate_artifact_binding
    validate_artifact_ref = _policy.validate_artifact_ref
    validate_policy_bundle = _policy.validate_policy_bundle

RECEIPT_KIND = "desired-results-policy-publication-receipt-v3"


class CreateOnlyStore(Protocol):
    """Minimal immutable object-store seam used by tests and the GCS adapter."""

    def create_only(self, uri: str, data: bytes) -> Mapping[str, Any]:
        """Create ``uri`` iff absent; return an ArtifactRef for its exact bytes."""


def _gs_parts(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if f"{parsed.scheme}://{parsed.netloc}{parsed.path}" != uri:
        raise ContractError(f"expected query-free gs://bucket/object URI, got {uri!r}")
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ContractError(f"expected canonical gs://bucket/object URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _destination(prefix: str, kind: str, digest: str) -> str:
    _gs_parts(prefix.rstrip("/") + "/sentinel")
    return f"{prefix.rstrip('/')}/{kind}/{digest}.json"


class GCSCreateOnlyStore:
    """Google Cloud Storage create-only adapter with same-byte idempotence."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:  # pragma: no cover - depends on production extra
                raise ContractError("google-cloud-storage is required with --execute") from exc
            client = storage.Client()
        self._client = client

    def create_only(self, uri: str, data: bytes) -> Mapping[str, Any]:
        bucket_name, object_name = _gs_parts(uri)
        blob = self._client.bucket(bucket_name).blob(object_name)
        try:
            blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
            generation = str(blob.generation)
        except Exception as exc:
            # google.api_core is optional in offline environments. Classify only
            # the documented create precondition failure; all other errors escape.
            code = getattr(exc, "code", None)
            if callable(code):
                code = code()
            if code not in (409, 412) and exc.__class__.__name__ not in {"Conflict", "PreconditionFailed"}:
                raise
            blob.reload()
            generation = str(blob.generation)
            pinned = self._client.bucket(bucket_name).blob(
                object_name, generation=int(generation),
            )
            existing = pinned.download_as_bytes(if_generation_match=int(generation))
            if existing != data:
                raise ContractError(f"immutable object conflict at {uri}") from exc
        if not generation.isdigit() or int(generation) < 1:
            raise ContractError(f"GCS returned a non-numeric generation for {uri}")
        digest = hashlib.sha256(data).hexdigest()
        ref = {
            "uri": uri,
            "generation": generation,
            "size": str(len(data)),
            "sha256": digest,
        }
        return validate_artifact_ref(ref, f"published {uri}")


def _create_exact(store: CreateOnlyStore, uri: str, payload: Any) -> dict[str, str]:
    data = canonical_json(payload)
    ref = validate_artifact_ref(store.create_only(uri, data), f"store result for {uri}")
    if ref["uri"] != uri or ref["size"] != str(len(data)) or ref["sha256"] != hashlib.sha256(data).hexdigest():
        raise ContractError(f"store result for {uri} does not bind the requested canonical bytes")
    if not ref["generation"].isdigit() or int(ref["generation"]) < 1:
        raise ContractError(f"store result for {uri} lacks a numeric immutable generation")
    return ref


def publication_plan(bundle: Mapping[str, Any], destination_prefix: str) -> dict[str, Any]:
    """Validate offline and return deterministic destinations without a network call."""
    validate_policy_bundle(bundle, allow_local_baselines=True)
    objects = []
    for binding in bundle["objects"]:
        ref = validate_artifact_binding(binding["ref"], binding["payload"], "bundle object")
        objects.append({
            "source_ref": ref,
            "destination_uri": _destination(destination_prefix, "baseline", ref["sha256"]),
            "sha256": ref["sha256"],
            "size": ref["size"],
        })
    objects.sort(key=lambda item: item["destination_uri"])
    plan_without_hash = {
        "schema_version": 3,
        "plan_kind": "desired-results-policy-publication-plan-v3",
        "source_bundle_sha256": bundle["bundle_sha256"],
        "destination_prefix": destination_prefix.rstrip("/"),
        "objects": objects,
    }
    plan = dict(plan_without_hash)
    plan["plan_sha256"] = canonical_sha256(plan_without_hash)
    return plan


def publish_policy_bundle(
    bundle: Mapping[str, Any],
    destination_prefix: str,
    *,
    store: CreateOnlyStore | None = None,
) -> dict[str, Any]:
    """Publish baseline objects, rewrite only their refs, and reseal policy/bundle."""
    plan = publication_plan(bundle, destination_prefix)
    # Reject non-production scientific inputs before the first network write.
    validate_policy_bundle(bundle, allow_local_baselines=True, production=True)
    if store is None:
        store = GCSCreateOnlyStore()

    original = copy.deepcopy(dict(bundle))
    published = copy.deepcopy(dict(bundle))
    replacement_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    new_objects: list[dict[str, Any]] = []
    for binding in published["objects"]:
        old_ref = validate_artifact_binding(binding["ref"], binding["payload"], "bundle object")
        uri = _destination(destination_prefix, "baseline", old_ref["sha256"])
        new_ref = _create_exact(store, uri, binding["payload"])
        key = (old_ref["uri"], old_ref["generation"], old_ref["size"], old_ref["sha256"])
        replacement_by_key[key] = new_ref
        new_objects.append({"ref": new_ref, "payload": binding["payload"]})
    new_objects.sort(key=lambda item: item["ref"]["uri"])

    for target_id, baseline_ref in published["policy"]["baselines"].items():
        old_ref = validate_artifact_ref(baseline_ref, f"baseline {target_id}")
        key = (old_ref["uri"], old_ref["generation"], old_ref["size"], old_ref["sha256"])
        if key not in replacement_by_key:
            raise ContractError(f"baseline {target_id} does not name a materializable bundle object")
        published["policy"]["baselines"][target_id] = replacement_by_key[key]
    published["objects"] = new_objects

    # These are the promoted scientific inputs.  Publication may not rewrite or
    # weaken any of them; only locally materializable baseline refs may change.
    if published["target_manifest_refs"] != original["target_manifest_refs"]:
        raise ContractError("publisher changed target_manifest_refs")
    if published["pair_text_refs"] != original["pair_text_refs"]:
        raise ContractError("publisher changed pair_text_refs")
    old_policy = original["policy"]
    new_policy = published["policy"]
    old_without_baselines = {key: value for key, value in old_policy.items() if key != "baselines"}
    new_without_baselines = {key: value for key, value in new_policy.items() if key != "baselines"}
    if old_without_baselines != new_without_baselines:
        raise ContractError("publisher changed policy fields other than baseline refs")

    published["policy_sha256"] = canonical_sha256(published["policy"])
    published_without_hash = dict(published)
    published_without_hash.pop("bundle_sha256", None)
    published["bundle_sha256"] = canonical_sha256(published_without_hash)
    validate_policy_bundle(published, allow_local_baselines=False)

    policy_uri = _destination(destination_prefix, "policy", published["policy_sha256"])
    policy_ref = _create_exact(store, policy_uri, published["policy"])
    bundle_uri = _destination(destination_prefix, "bundle", published["bundle_sha256"])
    bundle_ref = _create_exact(store, bundle_uri, published)
    receipt_without_hash = {
        "schema_version": 3,
        "receipt_kind": RECEIPT_KIND,
        "executed": True,
        "plan": plan,
        "published_bundle": published,
        "policy_ref": policy_ref,
        "bundle_ref": bundle_ref,
    }
    receipt = dict(receipt_without_hash)
    receipt["receipt_sha256"] = canonical_sha256(receipt_without_hash)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Local policy bundle JSON")
    parser.add_argument("--destination", required=True, help="Production gs:// bucket/prefix")
    parser.add_argument("--receipt", type=Path, required=True, help="Output execution receipt")
    parser.add_argument(
        "--execute", action="store_true", required=True,
        help="Acknowledge create-only production GCS writes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    receipt = publish_policy_bundle(bundle, args.destination)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_bytes(canonical_json(receipt) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
