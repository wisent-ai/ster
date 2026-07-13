#!/usr/bin/env python3
"""Explicitly publish an offline inventory plan and its descriptors to immutable GCS objects."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _run_v3() -> Any:
    path = Path(__file__).with_name("desired_results_run_v3.py")
    spec = importlib.util.spec_from_file_location("_desired_results_run_v3_publish", path)
    repo_root = str(path.resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load run-v3 module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_V3 = _run_v3()
PublishError = RUN_V3.RunV3Error


def _create_identical(store: Any, uri: str, data: bytes) -> dict[str, Any]:
    """Create *uri*, accepting an existing object only when its bytes are identical."""
    try:
        ref = store.create(uri, data)
    except Exception as create_error:
        resolve = getattr(store, "resolve", None)
        if resolve is None:
            raise PublishError(f"create-only publication failed for {uri}: {create_error}") from create_error
        ref = resolve(uri)
        if ref is None:
            raise PublishError(f"create-only publication failed for {uri}: {create_error}") from create_error
        try:
            existing = store.read(ref)
        except Exception as read_error:
            raise PublishError(f"cannot verify existing immutable object {uri}: {read_error}") from create_error
        if existing != data:
            raise PublishError(f"immutable publication conflict at {uri}") from create_error
    exact = RUN_V3._artifact_ref(ref, f"published object {uri}")
    if exact["uri"] != uri or exact["sha256"] != RUN_V3.hashlib.sha256(data).hexdigest() or exact["size"] != str(len(data)):
        raise PublishError(f"published object ref does not identify exact bytes at {uri}")
    return dict(exact)


def _validated_local_entries(bindings: Any, descriptor_dir: str | Path) -> list[tuple[Mapping[str, Any], bytes]]:
    """Bind each authenticated inventory path to its canonical content-addressed descriptor file."""
    root = Path(descriptor_dir).resolve()
    plan_entries = bindings.payload.get("descriptors")
    if not isinstance(plan_entries, list) or len(plan_entries) != len(bindings):
        raise PublishError("local inventory plan descriptor bindings are malformed")
    bindings_by_target = {
        binding.payload["target"]["target_id"]: binding
        for binding in bindings
    }
    validated: list[tuple[Mapping[str, Any], bytes]] = []
    for index, entry in enumerate(plan_entries):
        if not isinstance(entry, Mapping):
            raise PublishError(f"local inventory descriptor {index} must be an object")
        target_id = entry.get("target_id")
        binding = bindings_by_target.get(target_id)
        if binding is None:
            raise PublishError(f"local inventory descriptor {index} has no authenticated target")
        descriptor = binding.payload
        digest = descriptor["descriptor_sha256"]
        canonical_path = f"{digest}.json"
        if entry.get("descriptor_sha256") != digest or entry.get("path") != canonical_path:
            raise PublishError(f"local inventory descriptor {target_id} path is not canonical")
        path = (root / entry["path"]).resolve()
        if path.parent != root:
            raise PublishError(f"local inventory descriptor {target_id} path escapes descriptor_dir")
        raw = path.read_bytes()
        if RUN_V3.hashlib.sha256(raw).hexdigest() != binding.ref["sha256"] or str(len(raw)) != binding.ref["size"]:
            raise PublishError(f"local inventory descriptor {target_id} changed after validation")
        validated.append((descriptor, raw))
    return validated


def publish(inventory_plan_path: str | Path, descriptor_dir: str | Path,
            namespace: str, store: Any) -> dict[str, Any]:
    """Publish canonical descriptors and their non-circular immutable inventory plan."""
    if not isinstance(namespace, str) or not namespace.startswith("gs://"):
        raise PublishError("publication namespace must be a gs:// URI")
    namespace = namespace.rstrip("/")
    bindings = RUN_V3.load_inventory_plan(inventory_plan_path, descriptor_dir)
    descriptors: list[dict[str, Any]] = []
    for descriptor, raw in _validated_local_entries(bindings, descriptor_dir):
        digest = descriptor["descriptor_sha256"]
        uri = f"{namespace}/descriptors/{digest}.json"
        ref = _create_identical(store, uri, raw)
        descriptors.append({
            "target_id": descriptor["target"]["target_id"],
            "descriptor_sha256": digest,
            "descriptor_ref": ref,
        })
    descriptors.sort(key=lambda item: (item["descriptor_sha256"], item["target_id"]))
    remote_plan_without_hash = dict(bindings.payload)
    remote_plan_without_hash["descriptors"] = descriptors
    remote_plan_sha256 = RUN_V3.canonical_sha256(remote_plan_without_hash)
    remote_plan = dict(remote_plan_without_hash)
    remote_plan["plan_sha256"] = remote_plan_sha256
    plan_bytes = RUN_V3.canonical_bytes(remote_plan)
    plan_uri = f"{namespace}/inventory-plans/{remote_plan_sha256}.json"
    plan_ref = _create_identical(store, plan_uri, plan_bytes)
    return {
        "schema_version": 1,
        "inventory_plan": remote_plan,
        "inventory_plan_sha256": remote_plan_sha256,
        "inventory_plan_ref": plan_ref,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-plan", type=Path, required=True)
    parser.add_argument("--descriptor-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="perform create-only GCS publication (default only validates local inputs)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bindings = RUN_V3.load_inventory_plan(args.inventory_plan, args.descriptor_dir)
        _validated_local_entries(bindings, args.descriptor_dir)
        if not args.execute:
            result: Mapping[str, Any] = {
                "execute": False,
                "descriptor_count": len(bindings),
                "inventory_plan_sha256": bindings.inventory_plan_sha256,
            }
        else:
            if args.output is None:
                raise PublishError("--output is required with --execute")
            if args.output.exists():
                raise PublishError(f"publication receipt destination already exists: {args.output}")
            result = publish(
                args.inventory_plan, args.descriptor_dir, args.namespace, RUN_V3.GCSStore(),
            )
            RUN_V3.atomic_write_json(args.output, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        return 0
    except (PublishError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"inventory publication failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
