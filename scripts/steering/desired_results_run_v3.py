#!/usr/bin/env python3
"""Immutable desired-results v3 planner and lifecycle orchestrator.

The control plane is deliberately model-free.  It reads byte-exact artifacts,
derives content-addressed manifests, and dispatches only generation-pinned
production inputs.  Every mutating CLI phase is dry by default and requires an
explicit ``--execute`` gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.steering import desired_results_execution_contract as execution
    from scripts.steering import desired_results_selection
    from scripts.steering import desired_results_target
except (ImportError, ModuleNotFoundError):
    _MODULE_DIR = Path(__file__).resolve().parent
    def _load_local_module(name: str, filename: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, _MODULE_DIR / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {filename}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    execution = _load_local_module("desired_results_execution_contract_v3", "desired_results_execution_contract.py")
    desired_results_target = execution.desired_results_target
    desired_results_selection = _load_local_module(
        "desired_results_selection_v3", "desired_results_selection.py",
    )


SCHEMA_VERSION = execution.SCHEMA_VERSION
PILOT_TARGET_COUNT = 4
PRODUCTION_SCHEMES = ("gs://",)
LOCAL_SCHEMES = ("bundle://", "file://", "local://")
CALIBRATION_WORKER_COMMAND = ("python", "scripts/steering/desired_results_worker.py")
FINAL_TEST_WORKER_COMMAND = ("python", "scripts/steering/desired_results_final_test_worker.py")

def _policy_module() -> Any:
    try:
        from scripts.steering import desired_results_policy
        return desired_results_policy
    except (ImportError, ModuleNotFoundError):
        return _load_local_module("desired_results_policy_v3", "desired_results_policy.py")


def _final_test_module() -> Any:
    try:
        from scripts.steering import desired_results_final_test
        return desired_results_final_test
    except (ImportError, ModuleNotFoundError):
        return _load_local_module("desired_results_final_test_v3", "desired_results_final_test.py")


def _production_output_namespace(value: Any, label: str) -> str:
    try:
        return _policy_module().validate_production_output_namespace(value, label)
    except (execution.ContractError, ValueError, TypeError) as exc:
        raise RunV3Error(str(exc)) from exc


def _target_output_namespace(policy: Mapping[str, Any], target_id: str, phase: str,
                             method: str | None = None, *, production: bool = False) -> str:
    base = policy.get("output_namespace")
    if production:
        base = _production_output_namespace(base, "policy.output_namespace")
    elif not isinstance(base, str) or not base:
        raise RunV3Error("policy.output_namespace must be non-empty")
    target_key = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]
    if phase == "calibration" and isinstance(method, str) and method:
        return f"{base.rstrip('/')}/targets/{target_key}/calibration/{method}"
    if phase == "final" and method is None:
        return f"{base.rstrip('/')}/targets/{target_key}/final"
    raise RunV3Error("invalid target output namespace phase/method")


class RunV3Error(RuntimeError):
    """An immutable input, lifecycle transition, or dispatch is unsafe."""


def canonical_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Return canonical JSON bytes, optionally in the descriptor JSONL form."""
    try:
        data = execution.canonical_json(value)
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    return data + (b"\n" if trailing_newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strict_json(data: bytes, label: str, *, allow_trailing_newline: bool = False) -> Any:
    payload = data[:-1] if allow_trailing_newline and data.endswith(b"\n") else data
    if not payload or payload.endswith(b"\n"):
        raise RunV3Error(f"{label} has invalid trailing bytes")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunV3Error(f"{label} is not strict ASCII JSON: {exc}") from exc
    if canonical_bytes(value) != payload:
        raise RunV3Error(f"{label} is not canonical JSON")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    data = canonical_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
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


def _normalize_read(result: Any, expected_generation: str | None) -> tuple[bytes, str]:
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], bytes):
        raise RunV3Error("store.read must return (bytes, generation)")
    generation: Any = result[1]
    if isinstance(generation, Mapping):
        generation = generation.get("generation")
    if generation is None:
        raise RunV3Error("store.read did not return an object generation")
    generation = str(generation)
    if expected_generation is not None and generation != expected_generation:
        raise RunV3Error("store returned a different object generation")
    return result[0], generation


def _normalize_created(uri: str, data: bytes, result: Any) -> dict[str, str]:
    generation: Any = result.get("generation") if isinstance(result, Mapping) else result
    if generation is None:
        raise RunV3Error("store.create did not return an object generation")
    try:
        return execution.artifact_ref(uri, str(generation), str(len(data)), hashlib.sha256(data).hexdigest())
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc

def _artifact_ref(value: Any, label: str = "artifact") -> dict[str, str]:
    try:
        return execution.validate_artifact_ref(value, label)
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc


def is_production_ref(value: Mapping[str, Any]) -> bool:
    try:
        ref = execution.validate_artifact_ref(value)
    except execution.ContractError:
        return False
    return ref["uri"].startswith(PRODUCTION_SCHEMES) and ref["generation"].isdigit() and int(ref["generation"]) > 0


def require_production_ref(value: Mapping[str, Any], label: str = "artifact") -> dict[str, str]:
    try:
        ref = execution.validate_artifact_ref(value, label)
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    if not ref["uri"].startswith(PRODUCTION_SCHEMES):
        raise RunV3Error(f"{label} must use a production gs:// URI")
    if not ref["generation"].isdigit() or int(ref["generation"]) <= 0:
        raise RunV3Error(f"{label} must carry a positive numeric generation")
    return ref


def _artifact_refs(value: Any, path: str = "document") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if set(value) == set(execution.ARTIFACT_REF_KEYS):
            yield path, value
            return
        for key, child in value.items():
            yield from _artifact_refs(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _artifact_refs(child, f"{path}[{index}]")


def require_production_document(value: Any, label: str = "document") -> None:
    for path, ref in _artifact_refs(value, label):
        require_production_ref(ref, path)


def resolve_ref(store: Any, uri: str, label: str = "artifact") -> dict[str, str]:
    """Resolve an object to its exact current generation and byte identity."""
    if not isinstance(uri, str) or not uri:
        raise RunV3Error(f"{label} URI must be non-empty")
    if hasattr(store, "resolve"):
        result = store.resolve(uri)
        if isinstance(result, Mapping) and set(result) == set(execution.ARTIFACT_REF_KEYS):
            try:
                ref = execution.validate_artifact_ref(result, label)
            except execution.ContractError as exc:
                raise RunV3Error(str(exc)) from exc
            if ref["uri"] != uri:
                raise RunV3Error(f"{label} resolver returned a different URI")
            return ref
        if isinstance(result, tuple):
            data, generation = _normalize_read(result, None)
            return _normalize_created(uri, data, generation)
    data, generation = _normalize_read(store.read(uri, None), None)
    return _normalize_created(uri, data, generation)


def read_ref_bytes(store: Any, value: Mapping[str, Any], label: str = "artifact", *,
                   production: bool = False) -> tuple[bytes, dict[str, str]]:
    try:
        ref = execution.validate_artifact_ref(value, label)
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    if production:
        ref = require_production_ref(ref, label)
    data, generation = _normalize_read(store.read(ref["uri"], ref["generation"]), ref["generation"])
    if str(len(data)) != ref["size"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise RunV3Error(f"{label} bytes differ from their immutable ArtifactRef")
    return data, execution.artifact_ref(ref["uri"], generation, str(len(data)), hashlib.sha256(data).hexdigest())


def read_ref(store: Any, value: Mapping[str, Any], label: str = "artifact", *,
             production: bool = False, allow_trailing_newline: bool = False) -> tuple[Any, dict[str, str]]:
    data, ref = read_ref_bytes(store, value, label, production=production)
    return _strict_json(data, label, allow_trailing_newline=allow_trailing_newline), ref


def load_ref(store: Any, value: Mapping[str, Any], label: str = "artifact", **kwargs: Any) -> tuple[Any, dict[str, str]]:
    return read_ref(store, value, label, **kwargs)


def publish_bytes(store: Any, uri: str, data: bytes) -> dict[str, str]:
    """Create an object once; an identical existing object is idempotent."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise RunV3Error("publication URI must be a production gs:// URI")
    try:
        ref = _normalize_created(uri, data, store.create(uri, data))
    except Exception as create_error:
        try:
            observed, generation = _normalize_read(store.read(uri, None), None)
        except Exception:
            raise RunV3Error(f"create-only publication failed for {uri}: {create_error}") from create_error
        if observed != data:
            raise RunV3Error(f"conflicting object already exists at {uri}") from create_error
        ref = _normalize_created(uri, data, generation)
    observed, generation = _normalize_read(store.read(uri, ref["generation"]), ref["generation"])
    if observed != data:
        raise RunV3Error(f"created object failed immutable read-back at {uri}")
    return _normalize_created(uri, observed, generation)


def publish_json(store: Any, uri: str, value: Any) -> dict[str, str]:
    return publish_bytes(store, uri, canonical_bytes(value))


def _offline_ref(kind: str, value: Any) -> dict[str, str]:
    data = canonical_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    return execution.artifact_ref(
        f"bundle://desired-results-v3/{kind}/{digest}.json", digest, str(len(data)), digest
    )


class LocalStore:
    """Read-only local ArtifactRef adapter for offline planning."""

    @staticmethod
    def _path(uri: str) -> Path:
        if uri.startswith("file://"):
            return Path(uri[7:])
        if uri.startswith("local://"):
            return Path(uri[8:])
        raise RunV3Error("LocalStore accepts only file:// or local:// URIs")

    def read(self, uri: str, generation: str | None = None) -> tuple[bytes, str]:
        path = self._path(uri)
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if generation is not None and generation != observed:
            raise RunV3Error("local object generation differs")
        return data, observed

    def resolve(self, uri: str) -> dict[str, str]:
        data, generation = self.read(uri, None)
        return execution.artifact_ref(uri, generation, str(len(data)), hashlib.sha256(data).hexdigest())

    def create(self, uri: str, data: bytes) -> str:
        raise RunV3Error("LocalStore is read-only; offline planning never publishes")


class GCSStore:
    """Generation-pinned, create-only Google Cloud Storage adapter."""

    def __init__(self) -> None:
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RunV3Error("google-cloud-storage is required for --execute") from exc
        self.client = storage.Client()

    @staticmethod
    def _blob(client: Any, uri: str, generation: str | None = None) -> Any:
        if not uri.startswith("gs://"):
            raise RunV3Error("GCS URI must start with gs://")
        bucket, separator, name = uri[5:].partition("/")
        if not separator or not bucket or not name:
            raise RunV3Error("GCS URI must identify an object")
        pinned_generation = int(generation) if generation is not None else None
        return client.bucket(bucket).blob(name, generation=pinned_generation)

    def create(self, uri: str, data: bytes) -> dict[str, str]:
        blob = self._blob(self.client, uri)
        blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
        blob.reload()
        return execution.artifact_ref(uri, str(blob.generation), str(len(data)), hashlib.sha256(data).hexdigest())

    def read(self, uri: str | Mapping[str, Any], generation: str | None = None) -> Any:
        mapping_call = isinstance(uri, Mapping)
        if mapping_call:
            ref = _artifact_ref(uri)
            uri, generation = ref["uri"], ref["generation"]
        blob = self._blob(self.client, uri, generation)
        if generation is not None:
            data = blob.download_as_bytes(if_generation_match=int(generation))
            observed_generation = str(generation)
        else:
            blob.reload()
            observed_generation = str(blob.generation)
            data = blob.download_as_bytes(if_generation_match=int(observed_generation))
        return data if mapping_call else (data, observed_generation)

    def resolve(self, uri: str) -> dict[str, str]:
        data, generation = self.read(uri, None)
        return execution.artifact_ref(uri, generation, str(len(data)), hashlib.sha256(data).hexdigest())

def atomic_write_json(path: str | os.PathLike[str] | Path, value: Any) -> None:
    _atomic_json(Path(path), value)


def _validate_logical_hash(document: Mapping[str, Any], field: str, label: str) -> None:
    digest = document.get(field)
    if digest is None:
        return
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RunV3Error(f"{label}.{field} must be a lowercase SHA-256 digest")
    unhashed = dict(document)
    del unhashed[field]
    if canonical_sha256(unhashed) != digest:
        raise RunV3Error(f"{label}.{field} does not match its logical payload")


class InventoryBinding:
    def __init__(self, payload: Mapping[str, Any], ref: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        self.ref = dict(ref)

    def __getitem__(self, key: str) -> Any:
        if key == "payload":
            return self.payload
        if key == "ref":
            return self.ref
        raise KeyError(key)


class InventoryPlanBindings(Sequence[InventoryBinding]):
    def __init__(self, payload: Mapping[str, Any], ref: Mapping[str, Any],
                 bindings: Sequence[InventoryBinding], inventory_plan_sha256: str) -> None:
        self.payload = dict(payload)
        self.ref = dict(ref)
        self.bindings = tuple(bindings)
        self.inventory_plan_sha256 = inventory_plan_sha256

    def __len__(self) -> int:
        return len(self.bindings)

    def __getitem__(self, index: int) -> InventoryBinding:
        return self.bindings[index]


def _local_file_ref(path: Path, raw: bytes) -> dict[str, str]:
    digest = hashlib.sha256(raw).hexdigest()
    return execution.artifact_ref(f"file://{path.resolve()}", digest, str(len(raw)), digest)


def _load_local_inventory_plan(plan_path: Path, descriptor_dir: Path) -> InventoryPlanBindings:
    plan_path = Path(plan_path).resolve()
    descriptor_dir = Path(descriptor_dir).resolve()
    raw_plan = plan_path.read_bytes()
    value = _strict_json(raw_plan, "inventory plan", allow_trailing_newline=True)
    if not isinstance(value, Mapping) or not isinstance(value.get("descriptors"), list):
        raise RunV3Error("inventory plan is malformed")
    bindings: list[InventoryBinding] = []
    seen: set[str] = set()
    for index, entry in enumerate(value["descriptors"]):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise RunV3Error(f"inventory plan descriptor {index} lacks a local path")
        target_id = entry.get("target_id")
        if not isinstance(target_id, str) or not target_id or target_id in seen:
            raise RunV3Error("inventory plan target IDs must be unique non-empty strings")
        seen.add(target_id)
        descriptor_path = (descriptor_dir / entry["path"]).resolve()
        if descriptor_dir not in descriptor_path.parents:
            raise RunV3Error("inventory descriptor path escapes descriptor_dir")
        raw_descriptor = descriptor_path.read_bytes()
        descriptor = _strict_json(raw_descriptor, f"descriptor {target_id}", allow_trailing_newline=True)
        if not isinstance(descriptor, Mapping):
            raise RunV3Error(f"descriptor {target_id} must be an object")
        descriptor = dict(descriptor)
        _validate_logical_hash(descriptor, "descriptor_sha256", f"descriptor {target_id}")
        if descriptor.get("descriptor_sha256") != entry.get("descriptor_sha256"):
            raise RunV3Error(f"descriptor {target_id} logical hash differs from inventory plan")
        bindings.append(InventoryBinding(descriptor, _local_file_ref(descriptor_path, raw_descriptor)))
    if value.get("descriptor_count", len(bindings)) != len(bindings):
        raise RunV3Error("inventory plan descriptor_count differs")
    raw_sha = hashlib.sha256(raw_plan).hexdigest()
    return InventoryPlanBindings(value, _local_file_ref(plan_path, raw_plan), bindings, raw_sha)


def _load_inventory_plan_ref(store: Any, inventory_plan_ref: Mapping[str, Any], *,
                             production: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one byte-exact inventory plan without confusing raw and logical hashes."""
    value, ref = read_ref(store, inventory_plan_ref, "inventory plan", production=production,
                          allow_trailing_newline=True)
    if not isinstance(value, Mapping):
        raise RunV3Error("inventory plan must be an object")
    plan = dict(value)
    if plan.get("schema_version") not in {2, 3}:
        raise RunV3Error("inventory plan schema_version must be 2 or 3")
    descriptors = plan.get("descriptors")
    if not isinstance(descriptors, list):
        raise RunV3Error("inventory plan.descriptors must be a list")
    target_ids: set[str] = set()
    for index, entry in enumerate(descriptors):
        if not isinstance(entry, Mapping):
            raise RunV3Error(f"inventory plan.descriptors[{index}] must be an object")
        target_id = entry.get("target_id")
        if not isinstance(target_id, str) or not target_id or target_id in target_ids:
            raise RunV3Error("inventory plan descriptor target IDs must be non-empty and unique")
        target_ids.add(target_id)
        descriptor_sha = entry.get("descriptor_sha256")
        if not isinstance(descriptor_sha, str) or len(descriptor_sha) != 64:
            raise RunV3Error(f"inventory descriptor {target_id} lacks descriptor_sha256")
        descriptor_ref = entry.get("descriptor_ref")
        if descriptor_ref is not None:
            try:
                execution.validate_artifact_ref(descriptor_ref, f"inventory descriptor {target_id}.descriptor_ref")
            except execution.ContractError as exc:
                raise RunV3Error(str(exc)) from exc
    if "descriptor_count" in plan and plan["descriptor_count"] != len(descriptors):
        raise RunV3Error("inventory plan.descriptor_count differs from descriptors")
    _validate_logical_hash(plan, "plan_sha256", "inventory plan")
    return plan, ref


def load_inventory_plan(source: Any, inventory: Any, *, production: bool = False) -> Any:
    """Load either a local plan+descriptor directory or a published ArtifactRef plan."""
    if isinstance(source, (str, os.PathLike, Path)) and isinstance(inventory, (str, os.PathLike, Path)):
        return _load_local_inventory_plan(Path(source), Path(inventory))
    return _load_inventory_plan_ref(source, inventory, production=production)


def load_descriptor(store: Any, descriptor_ref: Mapping[str, Any], *,
                    expected_sha256: str | None = None, production: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    """Load canonical descriptor bytes and separately validate its logical self-hash."""
    raw, ref = read_ref_bytes(store, descriptor_ref, "descriptor", production=production)
    value = _strict_json(raw, "descriptor", allow_trailing_newline=True)
    if not isinstance(value, Mapping):
        raise RunV3Error("descriptor must be an object")
    descriptor = dict(value)
    _validate_logical_hash(descriptor, "descriptor_sha256", "descriptor")
    logical_sha = descriptor.get("descriptor_sha256")
    if expected_sha256 is not None and logical_sha != expected_sha256:
        raise RunV3Error("descriptor logical hash differs from inventory plan")
    return descriptor, ref


def load_policy(store: Any, policy_ref: Mapping[str, Any], *,
                production: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    value, ref = read_ref(store, policy_ref, "policy bundle", production=production,
                          allow_trailing_newline=True)
    if not isinstance(value, Mapping):
        raise RunV3Error("policy bundle must be an object")
    try:
        desired_results_policy = _policy_module()
        parameters = inspect.signature(desired_results_policy.validate_policy_bundle).parameters
        kwargs = {"allow_local_baselines": not production} if "allow_local_baselines" in parameters else {}
        desired_results_policy.validate_policy_bundle(value, **kwargs)
    except (execution.ContractError, desired_results_target.ContractError, ValueError) as exc:
        raise RunV3Error(f"invalid policy bundle: {exc}") from exc
    if production:
        require_production_document(value, "policy bundle")
    return dict(value), ref


def _validated_inventory_selection(
    selection: Mapping[str, Any], inventory_plan: Mapping[str, Any],
) -> tuple[list[str], str]:
    try:
        seal = desired_results_selection.validate_inventory_selection(
            selection, inventory_plan.get("inventory_sha256"),
        )
    except desired_results_selection.SelectionError as exc:
        raise RunV3Error(str(exc)) from exc
    return list(selection["target_ids"]), seal


def load_inventory_selection(store: Any, selection_ref: Mapping[str, Any], inventory_plan: Mapping[str, Any], *,
                             production: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    value, ref = read_ref(store, selection_ref, "inventory selection", production=production,
                          allow_trailing_newline=True)
    if not isinstance(value, Mapping):
        raise RunV3Error("inventory selection must be an object")
    selection = dict(value)
    target_ids, _ = _validated_inventory_selection(selection, inventory_plan)
    available = {entry["target_id"] for entry in inventory_plan["descriptors"]}
    missing = set(target_ids) - available
    if missing:
        raise RunV3Error(f"inventory selection contains unknown targets: {sorted(missing)}")
    return selection, ref


def select_inventory_targets(
    inventory_plan: Mapping[str, Any], selection: Mapping[str, Any] | None = None, *,
    full_default: bool = False,
) -> list[dict[str, Any]]:
    """Return either an explicitly authorized full inventory or an exact four-target pilot."""
    descriptors = inventory_plan.get("descriptors")
    if not isinstance(descriptors, list):
        raise RunV3Error("inventory plan.descriptors must be a list")
    if selection is None and not full_default:
        raise RunV3Error("inventory target selection requires a selection or explicit full_default=True")
    if selection is not None and full_default:
        raise RunV3Error("inventory selection and full_default are mutually exclusive")
    by_target = {entry["target_id"]: dict(entry) for entry in descriptors}
    if selection is None:
        return [by_target[target_id] for target_id in sorted(by_target)]
    target_ids, _ = _validated_inventory_selection(selection, inventory_plan)
    try:
        return [by_target[target_id] for target_id in target_ids]
    except KeyError as exc:
        raise RunV3Error(f"inventory selection contains unknown target {exc.args[0]}") from exc


def _policy_targets(bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    policy = bundle.get("policy")
    bindings = bundle.get("target_manifest_refs")
    if not isinstance(policy, Mapping) or not isinstance(bindings, list):
        raise RunV3Error("policy bundle target bindings are malformed")
    by_target: dict[str, Mapping[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {"ref", "payload"}:
            raise RunV3Error("policy target manifest binding is malformed")
        manifest = binding["payload"]
        try:
            desired_results_target.validate_target_manifest(manifest)
            execution.validate_artifact_binding(binding["ref"], manifest, "policy target_manifest_ref")
        except (desired_results_target.ContractError, execution.ContractError) as exc:
            raise RunV3Error(str(exc)) from exc
        target_id = manifest["target"]["target_id"]
        if target_id in by_target:
            raise RunV3Error(f"duplicate policy target {target_id}")
        by_target[target_id] = binding
    return policy, by_target


def _validate_planning_policy_bundle(policy_bundle: Mapping[str, Any]) -> Any:
    """Validate either canonical offline-local or fully published policy inputs."""
    desired_results_policy = _policy_module()
    errors: list[Exception] = []
    for allow_local_baselines in (True, False):
        try:
            desired_results_policy.validate_policy_bundle(
                policy_bundle, allow_local_baselines=allow_local_baselines,
            )
            return desired_results_policy
        except (execution.ContractError, desired_results_target.ContractError, ValueError) as exc:
            errors.append(exc)
    raise RunV3Error(f"invalid policy bundle: {errors[-1]}") from errors[-1]


def _validate_production_policy_bundle(policy_bundle: Mapping[str, Any]) -> Any:
    desired_results_policy = _policy_module()
    try:
        desired_results_policy.validate_policy_bundle(
            policy_bundle, allow_local_baselines=False, production=True,
        )
    except (execution.ContractError, desired_results_target.ContractError, ValueError, TypeError) as exc:
        raise RunV3Error(f"invalid production policy bundle: {exc}") from exc
    return desired_results_policy


def build_calibration_manifests(policy_bundle: Mapping[str, Any], *,
                                policy_ref: Mapping[str, Any] | None = None,
                                target_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Derive one self-contained pre-HPO CalibrationManifestV3 per target/method."""
    desired_results_policy = _validate_planning_policy_bundle(policy_bundle)
    policy, bindings = _policy_targets(policy_bundle)
    effective_policy_ref = dict(policy_ref) if policy_ref is not None else _offline_ref("policies", policy)
    try:
        execution.validate_artifact_binding(effective_policy_ref, policy, "policy_ref")
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    selected_ids = list(target_ids) if target_ids is not None else sorted(bindings)
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) - set(bindings):
        raise RunV3Error("calibration target_ids must be a unique subset of policy targets")
    manifests: list[dict[str, Any]] = []
    for target_id in selected_ids:
        binding = bindings[target_id]
        target_manifest = binding["payload"]
        source_target = target_manifest["target"]
        revisions = target_manifest["revisions"]
        for method in target_manifest["calibration"]["methods"]:
            calibration_policy = desired_results_policy.calibration_policy_for(
                policy, effective_policy_ref, target_id, method,
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "protocol": dict(target_manifest["protocol"]),
                "target": {key: source_target[key] for key in ("target_id", "model_name", "model_slug", "benchmark")},
                "target_manifest_ref": dict(binding["ref"]),
                "method": method,
                "revisions": {
                    "model": revisions["model_revision"],
                    "tokenizer": revisions["tokenizer_revision"],
                    "activation": revisions["activation_revision"],
                    "code": policy["revisions"]["code"],
                    "runtime": policy["revisions"]["runtime"],
                },
                "support": {
                    "train": list(target_manifest["support"]["splits"]["train"]),
                    "validation": list(target_manifest["support"]["splits"]["validation"]),
                },
                "activation_routes": [
                    {key: route[key] for key in ("strategy", "layer", "completion_ref", "proof_ref")}
                    for route in target_manifest["activation"]["routes"]
                ],
                "calibration_policy": calibration_policy,
                "evaluator": dict(policy["evaluator"]),
                "runtime": {
                    "revision": policy["revisions"]["runtime"],
                    "device": calibration_policy["options"]["device"],
                },
                "output_namespace": _target_output_namespace(policy, target_id, "calibration", method),
            }
            try:
                manifest = execution.finalize_calibration_manifest(payload)
                execution.validate_calibration_manifest(manifest, target_manifest)
            except execution.ContractError as exc:
                raise RunV3Error(f"cannot build calibration manifest for {target_id}/{method}: {exc}") from exc
            manifests.append(manifest)
    return manifests


def _calibration_attempt_prefix(manifest: Mapping[str, Any], attempt: int) -> str:
    return (f"{manifest['output_namespace'].rstrip('/')}/calibration-v3/"
            f"{manifest['manifest_sha256']}/attempt-{attempt}")


def build_calibration_attempts(manifests: Sequence[Mapping[str, Any]],
                               receipt_history: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Plan resumable, bounded calibration attempts without reading held-out test data."""
    if receipt_history is None:
        history: Mapping[str, Sequence[Mapping[str, Any]]] = {}
    elif not isinstance(receipt_history, Mapping):
        raise RunV3Error("receipt_history must be a manifest-id map")
    else:
        history = receipt_history
    manifest_ids = {
        manifest.get("manifest_id") for manifest in manifests if isinstance(manifest, Mapping)
    }
    unknown_history = set(history) - manifest_ids
    if unknown_history:
        raise RunV3Error("receipt_history contains an unknown or stale calibration manifest")
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        try:
            execution.validate_calibration_manifest(manifest)
        except execution.ContractError as exc:
            raise RunV3Error(str(exc)) from exc
        manifest_id = manifest["manifest_id"]
        if manifest_id in seen:
            raise RunV3Error("duplicate calibration manifest")
        seen.add(manifest_id)
        raw_receipts = history.get(manifest_id, ())
        if not isinstance(raw_receipts, Sequence) or isinstance(raw_receipts, (str, bytes)):
            raise RunV3Error(f"receipt history for {manifest_id} must be a sequence")
        receipts = list(raw_receipts)
        try:
            for receipt in receipts:
                execution.validate_calibration_receipt(receipt, manifest)
            decision = execution.calibration_retry_decision(receipts)
        except (execution.ContractError, AttributeError, TypeError) as exc:
            raise RunV3Error(f"invalid receipt history for current calibration manifest: {exc}") from exc
        if decision in {"complete", "failed"}:
            continue
        prior_attempt = max((receipt["attempt"] for receipt in receipts), default=0)
        attempt = prior_attempt + 1 if decision in {"claim", "retry"} else prior_attempt
        attempts.append({
            "target_id": manifest["target"]["target_id"],
            "method": manifest["method"],
            "manifest": dict(manifest),
            "manifest_ref": _offline_ref("calibration-manifests", manifest),
            "attempt": attempt,
            "attempt_id": execution.calibration_attempt_id(manifest["manifest_sha256"], attempt),
            "decision": decision,
            "dependencies": [],
            "output_namespace": _calibration_attempt_prefix(manifest, attempt),
        })
    return attempts


def _calibration_receipt_values(calibration_receipts: Any) -> list[Mapping[str, Any]]:
    values: list[Any] = []
    if isinstance(calibration_receipts, Mapping):
        for target_id, target_receipts in calibration_receipts.items():
            if not isinstance(target_id, str) or not target_id:
                raise RunV3Error("calibration receipt map keys must be target IDs")
            target_values = target_receipts.values() if isinstance(target_receipts, Mapping) else target_receipts
            if not isinstance(target_values, Iterable) or isinstance(target_values, (str, bytes)):
                raise RunV3Error(f"calibration receipts for {target_id} must be iterable")
            values.extend(target_values)
    elif isinstance(calibration_receipts, Sequence) and not isinstance(calibration_receipts, (str, bytes)):
        values.extend(calibration_receipts)
    else:
        raise RunV3Error("calibration_receipts must be a sequence or target map")
    receipts: list[Mapping[str, Any]] = []
    for receipt in values:
        if not isinstance(receipt, Mapping):
            raise RunV3Error("calibration receipt must be an object")
        try:
            execution.validate_calibration_success_receipt(receipt)
        except execution.ContractError as exc:
            raise RunV3Error(f"invalid calibration success receipt: {exc}") from exc
        receipts.append(receipt)
    return receipts


def _receipt_map(calibration_receipts: Any,
                 expected_manifests: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Bind receipts by the canonical manifest identity, never by caller-controlled labels."""
    expected_by_sha: dict[str, tuple[tuple[str, str], Mapping[str, Any]]] = {}
    for manifest in expected_manifests:
        key = (manifest["target"]["target_id"], manifest["method"])
        digest = manifest["manifest_sha256"]
        if digest in expected_by_sha:
            raise RunV3Error("canonical calibration manifests have duplicate identities")
        expected_by_sha[digest] = (key, manifest)
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in _calibration_receipt_values(calibration_receipts):
        expected = expected_by_sha.get(receipt["manifest_sha256"])
        if expected is None:
            raise RunV3Error("calibration receipt binds an undeclared canonical manifest")
        key, manifest = expected
        try:
            execution.validate_calibration_success_receipt(receipt, manifest)
        except execution.ContractError as exc:
            raise RunV3Error(f"calibration receipt does not bind its canonical manifest: {exc}") from exc
        if key in result:
            raise RunV3Error(f"duplicate calibration receipt for {key[0]}/{key[1]}")
        result[key] = receipt
    return result


def _execution_calibration_policy(policy: Mapping[str, Any], policy_ref: Mapping[str, Any],
                                  target_id: str, methods: Sequence[str]) -> dict[str, Any]:
    desired_results_policy = _policy_module()
    effective = [desired_results_policy.calibration_policy_for(policy, policy_ref, target_id, method)
                 for method in methods]
    first = effective[0]
    optimizer = first["options"]["optimizer"]
    common = {
        "name": first["name"], "version": first["version"],
        "policy_ref": first["policy_ref"], "device": first["options"]["device"],
        "backend": optimizer["backend"], "direction": optimizer["direction"],
        "seed": optimizer["seed"],
    }
    for item in effective[1:]:
        candidate = {
            "name": item["name"], "version": item["version"],
            "policy_ref": item["policy_ref"], "device": item["options"]["device"],
            "backend": item["options"]["optimizer"]["backend"],
            "direction": item["options"]["optimizer"]["direction"],
            "seed": item["options"]["optimizer"]["seed"],
        }
        if candidate != common:
            raise RunV3Error("execution contract requires common optimizer identity across methods")
    return {
        "name": common["name"], "version": common["version"],
        "policy_ref": common["policy_ref"],
        "options": {
            "device": common["device"],
            "optimizer": {
                "backend": common["backend"], "direction": common["direction"],
                "seed": common["seed"],
                "trials_per_strategy": {
                    method: item["options"]["optimizer"]["trials_per_strategy"]
                    for method, item in zip(methods, effective)
                },
                "method_space": {
                    method: item["options"]["optimizer"]["method_space"]
                    for method, item in zip(methods, effective)
                },
            },
        },
    }


def _raw_calibration_bindings(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    if isinstance(value, Mapping):
        if set(value) in ({"payload", "ref"}, set(execution.ARTIFACT_REF_KEYS)):
            return [value]
        flattened: list[Any] = []
        for target_id, target_values in value.items():
            if not isinstance(target_id, str) or not target_id:
                raise RunV3Error("calibration receipt map keys must be target IDs")
            if isinstance(target_values, Mapping) and set(target_values) not in ({"payload", "ref"}, set(execution.ARTIFACT_REF_KEYS)):
                flattened.extend(target_values.values())
            elif isinstance(target_values, Sequence) and not isinstance(target_values, (str, bytes)):
                flattened.extend(target_values)
            else:
                flattened.append(target_values)
        return flattened
    raise RunV3Error("calibration_receipts must be a sequence or target map")


def _validate_calibration_result(result: Mapping[str, Any], manifest: Mapping[str, Any],
                                 receipt: Mapping[str, Any]) -> None:
    expected_target = manifest["target"]
    checks = (
        (result.get("manifest_sha256"), manifest["manifest_sha256"], "manifest"),
        (result.get("target"), expected_target, "target"),
        (result.get("method"), manifest["method"], "method"),
        (result.get("revisions"), manifest["revisions"], "revisions"),
        (result.get("fit_support"), manifest["support"]["train"], "fit support"),
        (result.get("selection_support"), manifest["support"]["validation"], "selection support"),
        (result.get("runtime_evidence"), receipt["runtime_evidence"], "runtime evidence"),
        (result.get("selected_config"), receipt["selected_config"], "selected config"),
        (result.get("test_reads"), 0, "held-out test read count"),
        (result.get("test_pair_ids_read"), [], "held-out test identities"),
    )
    for observed, expected, label in checks:
        if observed != expected:
            raise RunV3Error(f"calibration result {label} differs from its manifest lifecycle")


def _calibration_receipt_uri(manifest: Mapping[str, Any], attempt: int, state: str) -> str:
    return f"{_calibration_attempt_prefix(manifest, attempt)}/{state}.json"


def _load_production_calibration_receipts(
    store: Any, calibration_receipts: Any,
    expected_manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_sha = {manifest["manifest_sha256"]: manifest for manifest in expected_manifests}
    loaded: list[dict[str, Any]] = []
    for index, raw in enumerate(_raw_calibration_bindings(calibration_receipts)):
        receipt, receipt_ref = _production_binding(store, raw, f"calibration success receipt {index}")
        manifest = expected_by_sha.get(receipt.get("manifest_sha256"))
        if manifest is None:
            raise RunV3Error("calibration receipt binds an undeclared canonical manifest")
        stored_manifest, manifest_ref = _production_document(
            store, receipt.get("manifest_ref"), f"calibration manifest {index}",
        )
        _same_canonical_payload(stored_manifest, manifest, f"calibration manifest {index}")
        expected_success_uri = _calibration_receipt_uri(manifest, receipt.get("attempt"), "success")
        if receipt_ref["uri"] != expected_success_uri:
            raise RunV3Error("calibration success receipt URI differs from its manifest attempt lifecycle")
        try:
            execution.validate_calibration_success_receipt(receipt, stored_manifest)
        except execution.ContractError as exc:
            raise RunV3Error(f"invalid calibration success receipt: {exc}") from exc
        previous: Mapping[str, Any] | None = None
        for state in ("claim", "prepared", "running"):
            phase, _ = _resolve_production_document(
                store, _calibration_receipt_uri(manifest, receipt["attempt"], state),
                f"calibration {state} receipt {index}",
            )
            try:
                execution.validate_calibration_receipt(phase, stored_manifest)
                if phase["state"] != state or phase["attempt"] != receipt["attempt"]:
                    raise execution.ContractError("phase path differs from receipt identity")
                if previous is not None:
                    execution.validate_calibration_transition(previous, phase)
            except execution.ContractError as exc:
                raise RunV3Error(f"invalid calibration lifecycle: {exc}") from exc
            previous = phase
        try:
            if previous is None:
                raise execution.ContractError("calibration lifecycle is incomplete")
            execution.validate_calibration_transition(previous, receipt)
        except execution.ContractError as exc:
            raise RunV3Error(f"invalid calibration lifecycle: {exc}") from exc
        result, normalized_result_ref = _production_document(
            store, receipt["result_ref"], f"calibration result {index}",
        )
        if normalized_result_ref != execution.validate_artifact_ref(receipt["result_ref"]):
            raise RunV3Error("calibration result ref changed during exact load")
        expected_result_uri = (f"{manifest['output_namespace'].rstrip('/')}/calibration-v3/"
                               f"{manifest['manifest_sha256']}/attempt-{receipt['attempt']}/result.json")
        if normalized_result_ref["uri"] != expected_result_uri:
            raise RunV3Error("calibration result URI differs from its manifest attempt lifecycle")
        _validate_calibration_result(result, stored_manifest, receipt)
        for path, ref in _artifact_refs(stored_manifest, "calibration manifest"):
            _verify_production_ref(store, ref, path)
        loaded.append(receipt)
    return loaded


def _load_production_policy_inputs(
    store: Any, policy_bundle: Mapping[str, Any], policy_ref: Mapping[str, Any] | None,
    evaluator_ref: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    if policy_ref is None or evaluator_ref is None:
        raise RunV3Error("executed sealing requires immutable policy_ref and evaluator_ref")
    policy, normalized_policy_ref = _production_document(store, policy_ref, "policy")
    evaluator, normalized_evaluator_ref = _production_document(store, evaluator_ref, "evaluator")
    _same_canonical_payload(policy, policy_bundle.get("policy"), "policy")
    _same_canonical_payload(evaluator, policy.get("evaluator"), "evaluator")
    verified = copy.deepcopy(policy_bundle)
    verified["policy"] = policy
    bindings = verified.get("target_manifest_refs")
    if not isinstance(bindings, list):
        raise RunV3Error("policy bundle target bindings are malformed")
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping) or set(binding) != {"ref", "payload"}:
            raise RunV3Error("policy target manifest binding is malformed")
        manifest, ref = _production_document(store, binding["ref"], f"policy target manifest {index}")
        _same_canonical_payload(manifest, binding["payload"], f"policy target manifest {index}")
        binding["payload"] = manifest
        binding["ref"] = ref
    try:
        _policy_module().validate_policy_bundle(verified, allow_local_baselines=False)
    except (execution.ContractError, desired_results_target.ContractError, ValueError) as exc:
        raise RunV3Error(f"invalid policy bundle: {exc}") from exc
    require_production_document(verified, "policy bundle")
    checked: set[tuple[str, str, str, str]] = set()
    for path, ref in _artifact_refs(verified, "policy bundle"):
        key = tuple(ref[field] for field in execution.ARTIFACT_REF_KEYS)
        if key not in checked:
            _verify_production_ref(store, ref, path)
            checked.add(key)
    return verified, normalized_policy_ref, normalized_evaluator_ref


def build_execution_contracts(policy_bundle: Mapping[str, Any], calibration_receipts: Any, *,
                              policy_ref: Mapping[str, Any] | None = None,
                              evaluator_ref: Mapping[str, Any] | None = None,
                              store: Any = None, production: bool = False) -> list[dict[str, Any]]:
    """Promote a complete canonical calibration set into target-isolated contracts."""
    if production:
        if store is None:
            raise RunV3Error("production execution-contract sealing requires store access")
        policy_bundle, effective_policy_ref, effective_evaluator_ref = _load_production_policy_inputs(
            store, policy_bundle, policy_ref, evaluator_ref,
        )
    else:
        effective_policy_ref = dict(policy_ref) if policy_ref is not None else _offline_ref("policies", policy_bundle["policy"])
        effective_evaluator_ref = (dict(evaluator_ref) if evaluator_ref is not None else
                                   _offline_ref("evaluators", policy_bundle["policy"]["evaluator"]))
    _validate_planning_policy_bundle(policy_bundle)
    policy, bindings = _policy_targets(policy_bundle)
    try:
        execution.validate_artifact_binding(effective_policy_ref, policy, "policy_ref")
        execution.validate_artifact_binding(effective_evaluator_ref, policy["evaluator"], "evaluator_ref")
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    expected_manifests = build_calibration_manifests(policy_bundle, policy_ref=effective_policy_ref)
    verified_receipts = (_load_production_calibration_receipts(store, calibration_receipts, expected_manifests)
                         if production else calibration_receipts)
    receipts = _receipt_map(verified_receipts, expected_manifests)
    evaluator = dict(policy["evaluator"])
    wave_runtime: tuple[str, Mapping[str, Any]] | None = None
    contracts: list[dict[str, Any]] = []
    declared: set[tuple[str, str]] = set()
    for target_id in sorted(bindings):
        binding = bindings[target_id]
        target_manifest = binding["payload"]
        methods = list(target_manifest["calibration"]["methods"])
        declared.update((target_id, method) for method in methods)
        if any((target_id, method) not in receipts for method in methods):
            raise RunV3Error(f"target {target_id} lacks complete canonical calibration receipts")
        selected_receipts = [receipts[(target_id, method)] for method in methods]
        runtime_evidence = selected_receipts[0]["runtime_evidence"]
        runtime_sha = selected_receipts[0]["runtime_evidence_sha256"]
        if any(receipt["runtime_evidence"] != runtime_evidence or receipt["runtime_evidence_sha256"] != runtime_sha
               for receipt in selected_receipts):
            raise RunV3Error(f"target {target_id} calibration receipts span different runtimes")
        shared_runtime = (runtime_evidence["runtime_revision"], runtime_evidence["packages"])
        if wave_runtime is None:
            wave_runtime = shared_runtime
        elif shared_runtime != wave_runtime:
            raise RunV3Error("sealed calibration receipts span different runtime revisions or packages across targets")
        source_target = target_manifest["target"]
        revisions = target_manifest["revisions"]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol": dict(target_manifest["protocol"]),
            "target_manifest": target_manifest,
            "target_manifest_ref": dict(binding["ref"]),
            "target": {key: source_target[key] for key in ("target_id", "model_name", "model_slug", "benchmark")},
            "revisions": {
                "model": revisions["model_revision"], "tokenizer": revisions["tokenizer_revision"],
                "activation": revisions["activation_revision"], "code": policy["revisions"]["code"],
                "runtime": policy["revisions"]["runtime"],
            },
            "matrix": {
                "strategies": list(target_manifest["calibration"]["strategies"]),
                "layers": list(range(1, target_manifest["calibration"]["layer_count"] + 1)),
                "methods": methods, "pairs": source_target["expected_pairs"],
                "splits": dict(target_manifest["support"]["split_counts"]),
            },
            "calibration_policy": _execution_calibration_policy(
                policy, effective_policy_ref, target_id, methods,
            ),
            "calibration_receipts": selected_receipts,
            "evaluator": evaluator,
            "evaluator_ref": effective_evaluator_ref,
            "final_test": {"split": "test", "evaluations_per_arm": 1},
            "arms": ["baseline", *methods],
            "retry_policy": {"max_pre_test_attempts": execution.MAX_PRE_TEST_ATTEMPTS},
            "output_namespace": _target_output_namespace(policy, target_id, "final"),
            "runtime_evidence": runtime_evidence,
            "runtime_evidence_sha256": runtime_sha,
        }
        try:
            contract = execution.finalize_execution_contract(payload)
        except execution.ContractError as exc:
            raise RunV3Error(f"cannot build execution contract for {target_id}: {exc}") from exc
        contracts.append(contract)
    extra = set(receipts) - declared
    if extra:
        raise RunV3Error(f"calibration receipts contain undeclared target/method entries: {sorted(extra)}")
    return contracts


def build_arm_manifests(contract: Mapping[str, Any], contract_ref: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive arm manifests; baseline never receives policy baseline config or support routes."""
    try:
        execution.validate_execution_contract(contract)
        execution.validate_artifact_binding(contract_ref, contract, "contract_ref")
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    manifests: dict[str, dict[str, Any]] = {}
    receipts = {receipt["selected_config"]["method"]: receipt for receipt in contract["calibration_receipts"]}
    routes = contract["target_manifest"]["activation"]["routes"]
    for arm in contract["arms"]:
        receipt = receipts.get(arm)
        if arm == "baseline":
            method = None
            selected_ref = None
            support_refs: list[dict[str, Any]] = []
        else:
            if receipt is None:
                raise RunV3Error(f"contract lacks calibration receipt for arm {arm}")
            method = arm
            selected_ref = receipt["result_ref"]
            selected = receipt["selected_config"]
            selected_routes = set(execution.selected_config_route_keys(selected))
            support_refs = [
                {key: route[key] for key in ("strategy", "layer", "completion_ref", "proof_ref")}
                for route in routes
                if (route["strategy"], route["layer"]) in selected_routes
            ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "contract_ref": dict(contract_ref),
            "contract_sha256": contract["contract_sha256"],
            "arm": arm,
            "method": method,
            "selected_config_ref": selected_ref,
            "pair_texts_ref": dict(contract["target_manifest"]["support"]["pair_texts_ref"]),
            "support_refs": support_refs,
            "evaluator_ref": dict(contract["evaluator_ref"]),
            "runtime_evidence": dict(contract["runtime_evidence"]),
            "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
            "output_namespace": contract["output_namespace"],
            "output_prefix": execution.derive_output_prefix(contract["output_namespace"], contract["contract_sha256"], arm),
        }
        try:
            manifest = execution.finalize_arm_manifest(payload)
            execution.validate_arm_manifest(manifest, contract)
        except execution.ContractError as exc:
            raise RunV3Error(f"cannot build arm manifest {arm}: {exc}") from exc
        manifests[arm] = manifest
    return manifests


def _production_binding(store: Any, value: Any, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Load canonical JSON from an immutable ref; caller payloads are comparison-only."""
    supplied: Any = None
    if isinstance(value, Mapping) and set(value) == set(execution.ARTIFACT_REF_KEYS):
        ref = value
    elif isinstance(value, Mapping) and set(value) == {"payload", "ref"}:
        supplied, ref = value["payload"], value["ref"]
        if not isinstance(supplied, Mapping):
            raise RunV3Error(f"{label} payload must be an object")
    else:
        raise RunV3Error(f"{label} must carry an immutable ArtifactRef")
    raw, normalized = read_ref_bytes(store, ref, label, production=True)
    document = _strict_json(raw, label)
    if not isinstance(document, Mapping):
        raise RunV3Error(f"{label} stored payload must be an object")
    if canonical_bytes(document) != raw:
        raise RunV3Error(f"{label} stored bytes are not canonical JSON")
    if supplied is not None and canonical_bytes(supplied) != raw:
        raise RunV3Error(f"{label} caller payload differs from immutable store bytes")
    return dict(document), normalized


def _production_document(store: Any, ref: Any, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    return _production_binding(store, ref, label)


def _verify_production_ref(store: Any, ref: Any, label: str) -> dict[str, str]:
    _, normalized = read_ref_bytes(store, ref, label, production=True)
    return normalized


def _resolve_production_document(store: Any, uri: str, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    return _production_document(store, resolve_ref(store, uri, label), label)


def _validate_production_route_artifacts(
    store: Any, manifest: Mapping[str, Any], label: str,
) -> None:
    """Exact-load content-addressed route evidence and bind it to its manifest route."""
    target = manifest["target"]
    target_id = target["target_id"]
    expected_pairs = target["expected_pairs"]
    for index, route in enumerate(manifest["activation"]["routes"]):
        route_label = f"{label} activation route {index}"
        completion, completion_ref = _production_document(
            store, route["completion_ref"], f"{route_label} completion",
        )
        proof, proof_ref = _production_document(
            store, route["proof_ref"], f"{route_label} proof",
        )
        if completion_ref != route["completion_ref"] or proof_ref != route["proof_ref"]:
            raise RunV3Error(f"{route_label} ref changed during exact load")
        expected_route = {"strategy": route["strategy"], "layer": route["layer"]}
        if (set(completion) != {
                "schema_version", "complete", "target_id", "route", "proof_ref",
                "activation_lfs_sha256", "activation_header_sha256",
        } or completion["schema_version"] != 2 or completion["complete"] is not True
                or completion["target_id"] != target_id or completion["route"] != expected_route
                or completion["proof_ref"] != proof_ref):
            raise RunV3Error(f"{route_label} completion differs from the manifest route")
        if (set(proof) != {
                "schema_version", "proof_kind", "target_id", "activation_artifact", "route",
                "pair_ids", "tensor_shapes", "tensor_dtypes", "safetensors_header_length",
                "safetensors_header_sha256", "tensor_payload_downloaded",
        } or proof["schema_version"] != 2
                or proof["proof_kind"] != "pinned_hf_safetensors_header"
                or proof["target_id"] != target_id or proof["route"] != expected_route
                or proof["tensor_payload_downloaded"] is not False):
            raise RunV3Error(f"{route_label} proof differs from the manifest route")
        artifact = proof["activation_artifact"]
        expected_path = (
            f"activations/{target['model_slug']}/{target['benchmark']}/"
            f"{route['strategy']}/layer_{route['layer']}.safetensors"
        )
        if (not isinstance(artifact, Mapping)
                or set(artifact) != {"repo_id", "repo_type", "revision", "path", "lfs_sha256", "size"}
                or artifact["revision"] != manifest["revisions"]["activation_revision"]
                or artifact["path"] != expected_path
                or artifact["lfs_sha256"] != completion["activation_lfs_sha256"]
                or not isinstance(artifact["size"], int) or artifact["size"] <= 0):
            raise RunV3Error(f"{route_label} activation artifact differs from target identity")
        if (proof["pair_ids"] != list(range(expected_pairs))
                or proof["safetensors_header_sha256"] != completion["activation_header_sha256"]
                or not isinstance(proof["safetensors_header_length"], int)
                or proof["safetensors_header_length"] <= 0):
            raise RunV3Error(f"{route_label} tensor support differs from target evidence")
        shapes, dtypes = proof["tensor_shapes"], proof["tensor_dtypes"]
        if (not isinstance(shapes, Mapping) or set(shapes) != {"pos_activations", "neg_activations"}
                or shapes["pos_activations"] != shapes["neg_activations"]
                or not isinstance(shapes["pos_activations"], list)
                or len(shapes["pos_activations"]) != 2
                or shapes["pos_activations"][0] != expected_pairs
                or not isinstance(shapes["pos_activations"][1], int)
                or shapes["pos_activations"][1] <= 0
                or not isinstance(dtypes, Mapping)
                or set(dtypes) != {"pos_activations", "neg_activations"}
                or dtypes["pos_activations"] != dtypes["neg_activations"]):
            raise RunV3Error(f"{route_label} tensor schema is inconsistent")


def _same_canonical_payload(observed: Mapping[str, Any], expected: Any, label: str) -> None:
    if not isinstance(expected, Mapping) or canonical_bytes(observed) != canonical_bytes(expected):
        raise RunV3Error(f"{label} differs from its immutable store payload")


PREFLIGHT_RECEIPT_KEYS = frozenset({
    "node_id", "status", "descriptor_sha256", "target_id", "inventory_plan_ref",
    "inventory_plan_sha256", "selection_ref", "selection_sha256", "submission_ref",
    "submission_sha256", "bundle_index_ref", "bundle_index", "completion_index_ref",
    "completion_index", "pair_texts_ref", "support_proof_ref", "target_manifest_ref",
    "target_manifest",
})


def _binding(value: Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, str]]:
    if isinstance(value, Mapping) and set(value) == {"payload", "ref"}:
        payload, ref = value["payload"], value["ref"]
    else:
        payload, ref = value, _offline_ref(label.replace(" ", "-"), value)
    if not isinstance(payload, Mapping):
        raise RunV3Error(f"{label} payload must be an object")
    try:
        execution.validate_artifact_binding(ref, payload, f"{label}.ref")
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    return dict(payload), dict(ref)


def promote(preflight_receipts: Sequence[Mapping[str, Any]], *,
            store: Any = None,
            inventory_plan_ref: Mapping[str, Any] | None = None,
            inventory_plan_sha256: str | None = None,
            selection_ref: Mapping[str, Any] | None = None,
            selection_sha256: str | None = None,
            submission_ref: Mapping[str, Any] | None = None,
            submission_sha256: str | None = None,
            production: bool = False) -> dict[str, Any]:
    """Promote terminal receipts only after byte-exact store verification."""
    if not isinstance(preflight_receipts, Sequence) or isinstance(preflight_receipts, (str, bytes)) or not preflight_receipts:
        raise RunV3Error("preflight_receipts must be a non-empty sequence")
    if production and store is None:
        raise RunV3Error("production promotion requires store access")
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    common_plan_ref: dict[str, str] | None = None
    common_plan_sha: str | None = None
    common_selection_ref: dict[str, str] | None = None
    common_selection_sha: str | None = None
    common_submission_ref: dict[str, str] | None = None
    common_submission_sha: str | None = None
    for index, raw_binding in enumerate(preflight_receipts):
        label = f"preflight receipt {index}"
        receipt, receipt_ref = (_production_binding(store, raw_binding, label)
                                if production else _binding(raw_binding, label))
        if set(receipt) != set(PREFLIGHT_RECEIPT_KEYS):
            raise RunV3Error(f"preflight receipt {index} has unexpected keys")
        if receipt["status"] != "complete":
            raise RunV3Error("only complete preflight receipts may be promoted")
        if production:
            require_production_document(receipt, label)
        target_id = receipt["target_id"]
        if not isinstance(target_id, str) or not target_id or target_id in seen:
            raise RunV3Error("preflight receipt target IDs must be unique non-empty strings")
        seen.add(target_id)

        if production:
            bundle, bundle_ref = _production_document(store, receipt["bundle_index_ref"], f"{label} bundle index")
            completion, completion_ref = _production_document(store, receipt["completion_index_ref"], f"{label} completion index")
            manifest, manifest_ref = _production_document(store, receipt["target_manifest_ref"], f"{label} target manifest")
            _same_canonical_payload(bundle, receipt["bundle_index"], f"{label} bundle index")
            _same_canonical_payload(completion, receipt["completion_index"], f"{label} completion index")
            _same_canonical_payload(manifest, receipt["target_manifest"], f"{label} target manifest")
            for ref_value, terminal_label in (
                (receipt["pair_texts_ref"], "pair texts"),
                (receipt["support_proof_ref"], "support proof"),
            ):
                _verify_production_ref(store, ref_value, f"{label} {terminal_label}")
        else:
            bundle, bundle_ref = dict(receipt["bundle_index"]), dict(receipt["bundle_index_ref"])
            completion, completion_ref = dict(receipt["completion_index"]), dict(receipt["completion_index_ref"])
            manifest, manifest_ref = dict(receipt["target_manifest"]), dict(receipt["target_manifest_ref"])
        try:
            execution.validate_artifact_binding(bundle_ref, bundle, "bundle_index_ref")
            execution.validate_artifact_binding(completion_ref, completion, "completion_index_ref")
            execution.validate_artifact_binding(manifest_ref, manifest, "target_manifest_ref")
            desired_results_target.validate_target_manifest(manifest)
        except (execution.ContractError, desired_results_target.ContractError) as exc:
            raise RunV3Error(f"invalid promoted target {target_id}: {exc}") from exc
        if production:
            if completion.get("routes") != manifest["activation"]["routes"]:
                raise RunV3Error(f"{label} completion route matrix differs from target manifest")
            _validate_production_route_artifacts(store, manifest, label)
        if (bundle.get("target_id") != target_id or bundle.get("descriptor_sha256") != receipt["descriptor_sha256"] or
                bundle.get("completion_index_ref") != completion_ref or bundle.get("target_manifest_ref") != manifest_ref):
            raise RunV3Error("bundle index differs from terminal receipt bindings")
        if (completion.get("target_id") != target_id or completion.get("complete") is not True or
                completion.get("pair_texts_ref") != receipt["pair_texts_ref"] or
                completion.get("support_proof_ref") != receipt["support_proof_ref"]):
            raise RunV3Error("completion index differs from terminal receipt bindings")
        if (manifest["target"]["target_id"] != target_id or
                manifest["support"]["pair_texts_ref"] != receipt["pair_texts_ref"] or
                manifest["support"]["state"] != "prepared"):
            raise RunV3Error("TargetManifestV2 is not the prepared target named by the receipt")

        plan_ref = _artifact_ref(receipt["inventory_plan_ref"], "inventory_plan_ref")
        selected_ref_value = receipt["selection_ref"]
        normalized_selection_ref = (None if selected_ref_value is None else
                                    _artifact_ref(selected_ref_value, "selection_ref"))
        submitted_ref = _artifact_ref(receipt["submission_ref"], "submission_ref")
        normalized_submission_sha = receipt["submission_sha256"]
        if production:
            inventory_plan, loaded_plan_ref = _load_inventory_plan_ref(store, plan_ref, production=True)
            submission_plan, loaded_submission_ref = load_preflight_plan(store, submitted_ref)
            if loaded_plan_ref != plan_ref or loaded_submission_ref != submitted_ref:
                raise RunV3Error("preflight source ref changed during exact load")
            logical_submission_sha = submission_plan["plan_sha256"]
            if receipt["submission_sha256"] not in {
                    logical_submission_sha, loaded_submission_ref["sha256"]}:
                raise RunV3Error("preflight receipt submission hash differs from exact submission bytes")
            normalized_submission_sha = logical_submission_sha
            if (inventory_plan.get("plan_sha256") != receipt["inventory_plan_sha256"]
                    or submission_plan["inventory_plan_ref"] != plan_ref
                    or submission_plan["inventory_plan_sha256"] != receipt["inventory_plan_sha256"]
                    or submission_plan["selection_ref"] != normalized_selection_ref
                    or submission_plan["selection_sha256"] != receipt["selection_sha256"]):
                raise RunV3Error("preflight receipt differs from authoritative submission lineage")
            matching_nodes = [item for item in submission_plan["targets"] if item["node_id"] == receipt["node_id"]]
            if (len(matching_nodes) != 1 or matching_nodes[0]["target_id"] != target_id or
                    matching_nodes[0]["descriptor_sha256"] != receipt["descriptor_sha256"]):
                raise RunV3Error("preflight receipt target differs from authoritative submission node")
            if normalized_selection_ref is not None:
                selection_document, loaded_selection_ref = load_inventory_selection(
                    store, normalized_selection_ref, inventory_plan, production=True,
                )
                observed_selection_sha = selection_document.get(
                    "selection_sha256", selection_document.get("content_sha256"),
                )
                if (loaded_selection_ref != normalized_selection_ref or
                        observed_selection_sha != receipt["selection_sha256"] or
                        target_id not in selection_document["target_ids"]):
                    raise RunV3Error("preflight receipt differs from authoritative inventory selection")
            elif receipt["selection_sha256"] is not None:
                raise RunV3Error("preflight receipt has a selection hash without an immutable selection_ref")
        identity = (
            plan_ref, receipt["inventory_plan_sha256"], normalized_selection_ref,
            receipt["selection_sha256"], submitted_ref, normalized_submission_sha,
        )
        if common_plan_ref is None:
            (common_plan_ref, common_plan_sha, common_selection_ref,
             common_selection_sha, common_submission_ref, common_submission_sha) = identity
        elif identity != (common_plan_ref, common_plan_sha, common_selection_ref,
                          common_selection_sha, common_submission_ref, common_submission_sha):
            raise RunV3Error("preflight receipts do not share one inventory/selection/submission seal")
        targets.append({
            "target_id": target_id,
            "preflight_receipt_ref": receipt_ref,
            "bundle_index_ref": bundle_ref,
            "target_manifest": manifest,
            "target_manifest_ref": manifest_ref,
            "pair_texts_ref": dict(receipt["pair_texts_ref"]),
            "support_proof_ref": dict(receipt["support_proof_ref"]),
        })
    for requested, observed, label in (
        (inventory_plan_ref, common_plan_ref, "inventory plan ref"),
        (selection_ref, common_selection_ref, "selection ref"),
        (submission_ref, common_submission_ref, "submission ref"),
    ):
        if requested is not None and _artifact_ref(requested, label) != observed:
            raise RunV3Error(f"promotion requested a different {label}")
    for requested, observed, label in (
        (inventory_plan_sha256, common_plan_sha, "inventory_plan_sha256"),
        (selection_sha256, common_selection_sha, "selection_sha256"),
        (submission_sha256, common_submission_sha, "submission_sha256"),
    ):
        if requested is not None and requested != observed:
            raise RunV3Error(f"promotion requested a different {label}")
    targets.sort(key=lambda item: item["target_id"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "promotion_kind": "desired-results-promotion-v3",
        "inventory_plan_ref": common_plan_ref,
        "inventory_plan_sha256": common_plan_sha,
        "selection_ref": common_selection_ref,
        "selection_sha256": common_selection_sha,
        "submission_ref": common_submission_ref,
        "submission_sha256": common_submission_sha,
        "targets": targets,
    }
    payload["promotion_sha256"] = canonical_sha256(payload)
    return payload


def promotion_policy_inputs(promotion: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return target bindings and target-scoped pair refs without creating a root pair map artifact."""
    targets = promotion.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RunV3Error("promotion targets are missing")
    bindings = [{"ref": target["target_manifest_ref"], "payload": target["target_manifest"]} for target in targets]
    pairs = {target["target_id"]: target["pair_texts_ref"] for target in targets}
    return bindings, pairs


def seal(store: Any, policy_bundle: Mapping[str, Any], calibration_receipts: Any, *,
         control_prefix: str, execute: bool = False,
         policy_ref: Mapping[str, Any] | None = None,
         evaluator_ref: Mapping[str, Any] | None = None, final_attempt: int = 1,
         submit_job_fn: Any = None) -> dict[str, Any]:
    """Seal contracts and arms only after every target method has calibrated."""
    if not isinstance(control_prefix, str) or not control_prefix:
        raise RunV3Error("control_prefix must be non-empty")
    if type(final_attempt) is not int or not 1 <= final_attempt <= execution.MAX_PRE_TEST_ATTEMPTS:
        raise RunV3Error(
            f"final_attempt must be an integer in [1, {execution.MAX_PRE_TEST_ATTEMPTS}]"
        )
    if execute:
        control_prefix = _production_output_namespace(control_prefix, "control_prefix")
        if policy_ref is None:
            raise RunV3Error("executed sealing requires immutable policy_ref")
        effective_policy_ref = require_production_ref(policy_ref, "policy_ref")
        if evaluator_ref is None:
            raise RunV3Error("executed sealing requires immutable evaluator_ref")
        effective_evaluator_ref = require_production_ref(evaluator_ref, "evaluator_ref")
    else:
        effective_policy_ref = (dict(policy_ref) if policy_ref is not None else
                                _offline_ref("policies", policy_bundle["policy"]))
        effective_evaluator_ref = (dict(evaluator_ref) if evaluator_ref is not None else
                                   _offline_ref("evaluators", policy_bundle["policy"]["evaluator"]))
    policy = policy_bundle["policy"]
    policy_sha = policy_bundle["policy_sha256"]
    try:
        _policy_module().validate_policy_bundle(policy_bundle, allow_local_baselines=not execute)
        if not execute:
            execution.validate_artifact_binding(effective_policy_ref, policy, "policy_ref")
            execution.validate_artifact_binding(effective_evaluator_ref, policy["evaluator"], "evaluator_ref")
    except (execution.ContractError, desired_results_target.ContractError, ValueError) as exc:
        raise RunV3Error(f"invalid policy bundle: {exc}") from exc
    contracts = build_execution_contracts(
        policy_bundle, calibration_receipts, policy_ref=effective_policy_ref,
        evaluator_ref=effective_evaluator_ref, store=store, production=execute,
    )
    sealed_targets: list[dict[str, Any]] = []
    for contract in contracts:
        target_id = contract["target"]["target_id"]
        target_key = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]
        target_prefix = f"{control_prefix.rstrip('/')}/targets/{target_key}"
        contract_uri = f"{target_prefix}/contracts/{contract['contract_sha256']}.json"
        contract_ref = publish_json(store, contract_uri, contract) if execute else _offline_ref("contracts", contract)
        manifests = build_arm_manifests(contract, contract_ref)
        manifest_refs: dict[str, dict[str, str]] = {}
        for arm, manifest in manifests.items():
            manifest_uri = f"{target_prefix}/arms/{arm}/{manifest['manifest_sha256']}.json"
            manifest_refs[arm] = publish_json(store, manifest_uri, manifest) if execute else _offline_ref("arm-manifests", manifest)
        seal_document = execution.finalize_final_seal({
            "schema_version": SCHEMA_VERSION,
            "contract": contract,
            "contract_ref": contract_ref,
            "contract_sha256": contract["contract_sha256"],
            "arm_manifest_refs": manifest_refs,
            "runtime_evidence": contract["runtime_evidence"],
            "runtime_evidence_sha256": contract["runtime_evidence_sha256"],
        })
        execution.validate_final_seal(seal_document, manifests)
        seal_uri = f"{target_prefix}/seals/{seal_document['seal_sha256']}.json"
        seal_ref = publish_json(store, seal_uri, seal_document) if execute else _offline_ref("final-seals", seal_document)
        if execute:
            require_production_document(seal_document, "final seal")
            require_production_ref(seal_ref, "final seal ref")
        sealed_targets.append({
            "target_id": target_id, "contract": contract, "contract_ref": contract_ref,
            "arm_manifests": manifests, "arm_manifest_refs": manifest_refs,
            "seal": seal_document, "seal_ref": seal_ref,
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "seal_kind": "desired-results-sealed-wave-v3",
        "policy_ref": effective_policy_ref,
        "policy_sha256": policy_sha,
        "targets": sealed_targets,
        "jobs": [],
        "submission_receipts": [],
    }
    if execute:
        result["jobs"] = build_final_jobs(policy_bundle, result, attempt=final_attempt)
        result["submission_receipts"] = _stado_module().dispatch_jobs(
            result["jobs"], submit=True, submit_job_fn=submit_job_fn,
        )
    result["seal_plan_sha256"] = canonical_sha256(result)
    return result


def _validate_final_result_document(result: Mapping[str, Any], contract: Mapping[str, Any],
                                    manifest: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    final_test = _final_test_module()
    try:
        validated = final_test.validate_staged_result(
            result, contract, manifest, f"final result {manifest.get('arm')}",
        )
    except final_test.FinalTestError as exc:
        raise RunV3Error(str(exc)) from exc
    if validated["test_token_id"] != attempt.get("test_token_id"):
        raise RunV3Error("final result test token differs from its sealed attempt graph")


def _load_production_seal_graph(
    store: Any, sealed_target: Mapping[str, Any], control_prefix: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(sealed_target, Mapping) or "seal_ref" not in sealed_target:
        raise RunV3Error("sealed target must carry an immutable seal_ref")
    seal, seal_ref = _production_document(store, sealed_target["seal_ref"], "final seal")
    contract, contract_ref = _production_document(store, seal.get("contract_ref"), "execution contract")
    _same_canonical_payload(contract, seal.get("contract"), "execution contract")
    manifests: dict[str, dict[str, Any]] = {}
    refs = seal.get("arm_manifest_refs")
    if not isinstance(refs, Mapping):
        raise RunV3Error("final seal arm_manifest_refs are malformed")
    for arm, ref in refs.items():
        manifest, _ = _production_document(store, ref, f"arm manifest {arm}")
        manifests[arm] = manifest
    require_production_document(seal, "final seal")
    require_production_document(contract, "execution contract")
    for arm, manifest in manifests.items():
        require_production_document(manifest, f"arm manifest {arm}")
    try:
        execution.validate_artifact_binding(contract_ref, contract, "final seal.contract_ref")
        execution.validate_final_seal(seal, manifests)
    except execution.ContractError as exc:
        raise RunV3Error(f"invalid immutable final seal graph: {exc}") from exc
    target_key = hashlib.sha256(contract["target"]["target_id"].encode("utf-8")).hexdigest()[:16]
    target_prefix = f"{control_prefix.rstrip('/')}/targets/{target_key}"
    if seal_ref["uri"] != f"{target_prefix}/seals/{seal['seal_sha256']}.json":
        raise RunV3Error("final seal ref is outside its canonical target namespace")
    if contract_ref["uri"] != f"{target_prefix}/contracts/{contract['contract_sha256']}.json":
        raise RunV3Error("execution contract ref is outside its canonical target namespace")
    for arm, manifest in manifests.items():
        expected_uri = f"{target_prefix}/arms/{arm}/{manifest['manifest_sha256']}.json"
        if refs[arm]["uri"] != expected_uri:
            raise RunV3Error("arm manifest ref is outside its canonical target namespace")
    return seal, seal_ref, contract, manifests


def _load_production_completion_graph(
    store: Any, raw: Mapping[str, Any], index: int, contract: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    completion, completion_ref = _production_binding(store, raw, f"completion {index}")
    require_production_document(completion, f"completion {index}")
    arm = completion.get("arm")
    manifest = manifests.get(arm) if isinstance(arm, str) else None
    if manifest is None:
        raise RunV3Error("completion names an arm outside the immutable final seal")
    attempt, attempt_ref = _production_document(
        store, completion.get("attempt_receipt_ref"), f"completed attempt {arm}",
    )
    require_production_document(attempt, f"completed attempt {arm}")
    try:
        execution.validate_completion_receipt(completion, contract, manifest, attempt)
    except execution.ContractError as exc:
        raise RunV3Error(f"invalid completion for {arm}: {exc}") from exc
    attempt_number = attempt["attempt"]
    expected_prefix = f"{manifest['output_prefix']}attempts/{attempt_number}/"
    expected_paths = {
        "completion": f"{manifest['output_prefix']}completion.json",
        "attempt": f"{expected_prefix}completed.json",
        "staged": f"{expected_prefix}staged-result.json",
        "publication": f"{manifest['output_prefix']}result.json",
    }
    if (completion_ref["uri"] != expected_paths["completion"] or
            attempt_ref["uri"] != expected_paths["attempt"]):
        raise RunV3Error("completion/attempt URI differs from the sealed lifecycle namespace")
    staged_raw, staged_ref = read_ref_bytes(
        store, completion["staged_result_ref"], f"staged result {arm}", production=True,
    )
    publication_raw, publication_ref = read_ref_bytes(
        store, completion["publication_ref"], f"published result {arm}", production=True,
    )
    if staged_ref["uri"] != expected_paths["staged"] or publication_ref["uri"] != expected_paths["publication"]:
        raise RunV3Error("staged/publication URI differs from the sealed lifecycle namespace")
    if staged_raw != publication_raw:
        raise RunV3Error("published result differs from the exact durable staged result")
    staged = _strict_json(staged_raw, f"staged result {arm}")
    if not isinstance(staged, Mapping) or canonical_bytes(staged) != staged_raw:
        raise RunV3Error(f"staged result {arm} is not a canonical JSON object")
    _validate_final_result_document(staged, contract, manifest, attempt)
    return completion, completion_ref


def finalize(store: Any, sealed_target: Mapping[str, Any],
             completion_bindings: Sequence[Mapping[str, Any]], *,
             control_prefix: str, execute: bool = False) -> dict[str, Any]:
    """Finalize one target only after loading the complete immutable store graph."""
    if not isinstance(control_prefix, str) or not control_prefix:
        raise RunV3Error("control_prefix must be non-empty")
    if not isinstance(completion_bindings, Sequence) or isinstance(completion_bindings, (str, bytes)):
        raise RunV3Error("completion_bindings must be a sequence")
    if execute:
        control_prefix = _production_output_namespace(control_prefix, "control_prefix")
        seal, seal_ref, contract, manifests = _load_production_seal_graph(
            store, sealed_target, control_prefix,
        )
    else:
        contract = sealed_target.get("contract")
        manifests = sealed_target.get("arm_manifests")
        seal = sealed_target.get("seal")
        seal_ref = sealed_target.get("seal_ref")
        if not isinstance(contract, Mapping) or not isinstance(manifests, Mapping):
            raise RunV3Error("sealed target lacks contract/arm manifests")
        try:
            execution.validate_final_seal(seal, manifests)
            execution.validate_artifact_binding(seal_ref, seal, "seal_ref")
        except execution.ContractError as exc:
            raise RunV3Error(str(exc)) from exc
    completions: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for index, raw in enumerate(completion_bindings):
        if execute:
            completion, ref = _load_production_completion_graph(store, raw, index, contract, manifests)
        else:
            completion, ref = _binding(raw, f"completion {index}")
            arm_value = completion.get("arm")
            try:
                execution.validate_completion_receipt(completion, contract, manifests.get(arm_value))
            except execution.ContractError as exc:
                raise RunV3Error(f"invalid completion for {arm_value}: {exc}") from exc
        arm = completion.get("arm")
        if not isinstance(arm, str) or arm in completions:
            raise RunV3Error("completion arms must be unique non-empty strings")
        completions[arm] = (completion, ref)
    if set(completions) != set(contract["arms"]):
        raise RunV3Error("finalizer requires exactly one completion for every sealed arm")
    target_key = hashlib.sha256(contract["target"]["target_id"].encode("utf-8")).hexdigest()[:16]
    prefix = f"{control_prefix.rstrip('/')}/targets/{target_key}/finalization"
    final_result = {
        "schema_version": SCHEMA_VERSION,
        "result_kind": "desired-results-final-result-v3",
        "contract_sha256": contract["contract_sha256"],
        "target": dict(contract["target"]),
        "arms": {
            arm: {"completion_ref": completions[arm][1],
                  "publication_ref": completions[arm][0]["publication_ref"]}
            for arm in contract["arms"]
        },
    }
    result_sha = canonical_sha256(final_result)
    result_uri = f"{prefix}/results/{result_sha}.json"
    result_ref = publish_json(store, result_uri, final_result) if execute else _offline_ref("final-results", final_result)
    receipt = execution.finalize_finalization_receipt({
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "seal_ref": seal_ref,
        "completion_refs": {arm: completions[arm][1] for arm in contract["arms"]},
        "final_result_ref": result_ref,
    })
    execution.validate_finalization_receipt(receipt, contract, [item[0] for item in completions.values()])
    receipt_uri = f"{prefix}/receipts/{receipt['finalization_sha256']}.json"
    receipt_ref = publish_json(store, receipt_uri, receipt) if execute else _offline_ref("finalization-receipts", receipt)
    if execute:
        require_production_ref(result_ref, "final result ref")
        require_production_ref(receipt_ref, "finalization receipt ref")
    return {"result": final_result, "result_ref": result_ref,
            "finalization": receipt, "finalization_ref": receipt_ref}


def _stado_module() -> Any:
    try:
        from scripts.steering import desired_results_stado
        return desired_results_stado
    except (ImportError, ModuleNotFoundError):
        return _load_local_module("desired_results_stado_v3", "desired_results_stado.py")


def plan(policy_bundle: Mapping[str, Any], *, policy_ref: Mapping[str, Any] | None = None,
         target_ids: Sequence[str] | None = None,
         receipt_history: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Build an offline-safe calibration plan; no held-out test rows or commands are copied."""
    policy = policy_bundle["policy"]
    effective_policy_ref = dict(policy_ref) if policy_ref is not None else _offline_ref("policies", policy)
    manifests = build_calibration_manifests(
        policy_bundle, policy_ref=effective_policy_ref, target_ids=target_ids,
    )
    attempts = build_calibration_attempts(manifests, receipt_history)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_kind": "desired-results-calibration-plan-v3",
        "policy_ref": effective_policy_ref,
        "policy_sha256": policy_bundle["policy_sha256"],
        "attempts": attempts,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def _execution_profile_for(policy: Mapping[str, Any], model_name: str, phase: str) -> dict[str, Any]:
    try:
        resource = _policy_module().resource_for(policy, model_name, phase)
        resource["dependency_lock_ref"] = require_production_ref(
            resource["dependency_lock_ref"], "policy resource dependency_lock_ref",
        )
    except (execution.ContractError, ValueError, KeyError, TypeError) as exc:
        raise RunV3Error(f"policy has no exact execution profile for {model_name}/{phase}: {exc}") from exc
    return resource


def build_calibration_jobs(policy_bundle: Mapping[str, Any],
                           attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Translate policy-bound, published calibration attempts to exact Stado jobs."""
    _validate_production_policy_bundle(policy_bundle)
    stado = _stado_module()
    policy = policy_bundle["policy"]
    policy_sha = policy_bundle["policy_sha256"]
    jobs: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or not isinstance(attempt.get("manifest"), Mapping):
            raise RunV3Error("calibration attempt must retain its exact manifest")
        manifest = attempt["manifest"]
        target = manifest.get("target")
        target_id = target.get("target_id") if isinstance(target, Mapping) else None
        method = manifest.get("method")
        if attempt.get("target_id") != target_id or attempt.get("method") != method:
            raise RunV3Error("calibration attempt target/method differs from its manifest")
        calibration_policy = manifest.get("calibration_policy")
        policy_ref = calibration_policy.get("policy_ref") if isinstance(calibration_policy, Mapping) else None
        expected_manifests = build_calibration_manifests(
            policy_bundle, policy_ref=policy_ref, target_ids=[target_id],
        )
        expected_manifest = next((item for item in expected_manifests if item["method"] == method), None)
        if expected_manifest is None or manifest != expected_manifest:
            raise RunV3Error("calibration manifest differs from the exact current production policy")
        expected_namespace = _target_output_namespace(
            policy, target_id, "calibration", method, production=True,
        )
        if manifest["output_namespace"] != expected_namespace:
            raise RunV3Error("calibration manifest output_namespace differs from current policy")
        attempt_number = attempt.get("attempt")
        try:
            expected_attempt_id = execution.calibration_attempt_id(manifest["manifest_sha256"], attempt_number)
        except (execution.ContractError, TypeError) as exc:
            raise RunV3Error(f"calibration attempt number is invalid: {exc}") from exc
        if attempt.get("attempt_id") != expected_attempt_id:
            raise RunV3Error("calibration attempt_id differs from its manifest")
        output_namespace = _calibration_attempt_prefix(manifest, attempt_number)
        _production_output_namespace(output_namespace, "calibration attempt output_namespace")
        if attempt.get("output_namespace") != output_namespace:
            raise RunV3Error("calibration attempt output_namespace differs from current policy")
        profile = _execution_profile_for(policy, manifest["target"]["model_name"], "calibration")
        runtime = {"package": stado.RUNTIME_PACKAGE, "device": manifest["runtime"]["device"],
                   "revision": policy["revisions"]["runtime"]}
        manifest_ref = require_production_ref(attempt["manifest_ref"], "calibration manifest_ref")
        try:
            execution.validate_artifact_binding(manifest_ref, manifest, "calibration manifest_ref")
        except execution.ContractError as exc:
            raise RunV3Error(str(exc)) from exc
        command = [
            *CALIBRATION_WORKER_COMMAND,
            "--calibration-manifest", manifest_ref["uri"],
            "--calibration-manifest-generation", manifest_ref["generation"],
            "--attempt-number", str(attempt_number),
        ]
        input_refs: list[Mapping[str, Any]] = [
            manifest_ref, manifest["target_manifest_ref"], manifest["calibration_policy"]["policy_ref"],
        ]
        target_manifest = next(
            binding["payload"] for binding in policy_bundle["target_manifest_refs"]
            if binding["payload"]["target"]["target_id"] == target_id
        )
        input_refs.append(target_manifest["support"]["pair_texts_ref"])
        for route in manifest["activation_routes"]:
            input_refs.extend((route["completion_ref"], route["proof_ref"]))
        job = stado.build_job(
            phase="calibration", command=command, policy_sha256=policy_sha,
            code_commit=manifest["revisions"]["code"],
            resources={key: profile[key] for key in ("accelerator", "memory_bytes", "runtime_seconds")},
            image=profile["image"], dependency_lock_ref=profile["dependency_lock_ref"],
            secrets=[], runtime=runtime,
            output_prefix=output_namespace, input_refs=input_refs,
            dependencies=attempt.get("dependencies", ()),
            manifest_ref=manifest_ref, attempt=attempt_number,
            attempt_id=attempt["attempt_id"], target_id=target_id,
            method=method,
        )
        jobs.append(job)
    return jobs


CALIBRATION_PLAN_KEYS = frozenset({
    "schema_version", "plan_kind", "policy_ref", "policy_sha256", "attempts", "plan_sha256",
})
CALIBRATION_PLAN_ATTEMPT_KEYS = frozenset({
    "target_id", "method", "manifest", "manifest_ref", "attempt", "attempt_id",
    "decision", "dependencies", "output_namespace",
})


def _validate_submission_plan(policy_bundle: Mapping[str, Any],
                              run_plan: Mapping[str, Any]) -> str:
    """Re-derive every policy-dependent plan field before publication."""
    _validate_planning_policy_bundle(policy_bundle)
    if not isinstance(run_plan, Mapping) or set(run_plan) != CALIBRATION_PLAN_KEYS:
        raise RunV3Error("calibration plan fields are malformed")
    if run_plan["schema_version"] != SCHEMA_VERSION or run_plan["plan_kind"] != "desired-results-calibration-plan-v3":
        raise RunV3Error("submit requires a calibration plan")
    unhashed = dict(run_plan)
    supplied_sha = unhashed.pop("plan_sha256")
    if supplied_sha != canonical_sha256(unhashed):
        raise RunV3Error("calibration plan_sha256 mismatch")
    policy = policy_bundle["policy"]
    if run_plan["policy_sha256"] != policy_bundle["policy_sha256"]:
        raise RunV3Error("calibration plan policy_sha256 differs from policy bundle")
    try:
        execution.validate_artifact_binding(run_plan["policy_ref"], policy, "calibration plan policy_ref")
    except execution.ContractError as exc:
        raise RunV3Error(str(exc)) from exc
    raw_attempts = run_plan["attempts"]
    if not isinstance(raw_attempts, list):
        raise RunV3Error("calibration plan attempts must be a list")
    target_ids: list[str] = []
    for item in raw_attempts:
        if not isinstance(item, Mapping) or set(item) != CALIBRATION_PLAN_ATTEMPT_KEYS:
            raise RunV3Error("calibration plan attempt fields are malformed")
        target_id = item["target_id"]
        if not isinstance(target_id, str) or not target_id:
            raise RunV3Error("calibration plan attempt target_id is invalid")
        method = item["method"]
        if not isinstance(method, str) or not method:
            raise RunV3Error("calibration plan attempt method is invalid")
        if target_id not in target_ids:
            target_ids.append(target_id)
    expected_manifests = build_calibration_manifests(
        policy_bundle, policy_ref=run_plan["policy_ref"], target_ids=target_ids,
    )
    expected_by_key = {
        (manifest["target"]["target_id"], manifest["method"]): (index, manifest)
        for index, manifest in enumerate(expected_manifests)
    }
    seen: set[tuple[str, str]] = set()
    previous_index = -1
    for item in raw_attempts:
        key = (item["target_id"], item["method"])
        expected = expected_by_key.get(key)
        if expected is None:
            raise RunV3Error("calibration plan attempt is outside its canonical target subset")
        if key in seen:
            raise RunV3Error(f"duplicate calibration plan attempt for {key[0]}/{key[1]}")
        seen.add(key)
        index, manifest = expected
        if index <= previous_index:
            raise RunV3Error("calibration plan attempts are not in canonical target/method order")
        previous_index = index
        if item["manifest"] != manifest:
            raise RunV3Error(f"calibration plan manifest differs from policy for {key[0]}/{key[1]}")
        if item["manifest_ref"] != _offline_ref("calibration-manifests", manifest):
            raise RunV3Error(f"calibration plan manifest_ref differs for {key[0]}/{key[1]}")
        attempt = item["attempt"]
        try:
            expected_attempt_id = execution.calibration_attempt_id(manifest["manifest_sha256"], attempt)
        except (execution.ContractError, TypeError) as exc:
            raise RunV3Error(f"calibration plan attempt number is invalid: {exc}") from exc
        if item["attempt_id"] != expected_attempt_id:
            raise RunV3Error(f"calibration plan attempt_id differs for {key[0]}/{key[1]}")
        decision = item["decision"]
        if decision not in {"claim", "retry", "resume"}:
            raise RunV3Error("calibration plan attempt decision is invalid")
        if (decision == "claim" and attempt != 1) or (decision == "retry" and attempt <= 1):
            raise RunV3Error("calibration plan attempt decision contradicts its attempt number")
        if item["dependencies"] != []:
            raise RunV3Error("calibration plan attempt dependencies differ from canonical plan")
        expected_namespace = _calibration_attempt_prefix(manifest, attempt)
        if item["output_namespace"] != expected_namespace:
            raise RunV3Error(f"calibration plan output_namespace differs for {key[0]}/{key[1]}")
    return supplied_sha


def submit(store: Any, policy_bundle: Mapping[str, Any], run_plan: Mapping[str, Any], *,
           control_prefix: str, execute: bool = False,
           submit_job_fn: Any = None) -> dict[str, Any]:
    """Publish exact manifests and optionally submit; default performs no write or dispatch."""
    supplied_sha = _validate_submission_plan(policy_bundle, run_plan)
    if not execute:
        return {"execute": False, "plan_sha256": supplied_sha,
                "attempt_count": len(run_plan["attempts"]), "jobs": [], "receipts": []}
    _validate_production_policy_bundle(policy_bundle)
    control_prefix = _production_output_namespace(control_prefix, "control_prefix")
    require_production_document(policy_bundle, "policy bundle")
    for item in run_plan["attempts"]:
        manifest = item["manifest"]
        _execution_profile_for(policy_bundle["policy"], manifest["target"]["model_name"], "calibration")
    attempts: list[dict[str, Any]] = []
    for item in run_plan["attempts"]:
        attempt = dict(item)
        manifest = attempt["manifest"]
        target_key = hashlib.sha256(attempt["target_id"].encode("utf-8")).hexdigest()[:16]
        uri = (f"{control_prefix.rstrip('/')}/targets/{target_key}/calibration/"
               f"{attempt['method']}/manifests/{manifest['manifest_sha256']}.json")
        attempt["manifest_ref"] = publish_json(store, uri, manifest)
        attempts.append(attempt)
    jobs = build_calibration_jobs(policy_bundle, attempts)
    receipts = _stado_module().dispatch_jobs(jobs, submit=True, submit_job_fn=submit_job_fn)
    return {"execute": True, "plan_sha256": supplied_sha, "attempts": attempts,
            "jobs": jobs, "receipts": receipts}


def build_final_jobs(policy_bundle: Mapping[str, Any], sealed_plan: Mapping[str, Any], *,
                     attempt: int = 1, dependencies: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Build one policy-bound Stado job per immutable arm manifest."""
    if type(attempt) is not int or not 1 <= attempt <= execution.MAX_PRE_TEST_ATTEMPTS:
        raise RunV3Error("final attempt is outside retry policy")
    _validate_production_policy_bundle(policy_bundle)
    stado = _stado_module()
    policy = policy_bundle["policy"]
    expected_targets = {
        binding["payload"]["target"]["target_id"]: binding
        for binding in policy_bundle["target_manifest_refs"]
    }
    jobs: list[dict[str, Any]] = []
    for target in sealed_plan["targets"]:
        if not isinstance(target, Mapping) or not isinstance(target.get("contract"), Mapping):
            raise RunV3Error("sealed target must retain its exact execution contract")
        contract = target["contract"]
        try:
            execution.validate_execution_contract(contract)
        except execution.ContractError as exc:
            raise RunV3Error(f"invalid final execution contract: {exc}") from exc
        target_id = contract["target"]["target_id"]
        binding = expected_targets.get(target_id)
        if (target.get("target_id") != target_id or binding is None or
                contract["target_manifest"] != binding["payload"] or
                contract["target_manifest_ref"] != binding["ref"]):
            raise RunV3Error("final execution contract target differs from current policy")
        expected_namespace = _target_output_namespace(policy, target_id, "final", production=True)
        if contract["output_namespace"] != expected_namespace:
            raise RunV3Error("final execution contract output_namespace differs from current policy")
        policy_ref = contract["calibration_policy"]["policy_ref"]
        try:
            execution.validate_artifact_binding(policy_ref, policy, "final contract policy_ref")
        except execution.ContractError as exc:
            raise RunV3Error(str(exc)) from exc
        methods = binding["payload"]["calibration"]["methods"]
        if contract["calibration_policy"] != _execution_calibration_policy(policy, policy_ref, target_id, methods):
            raise RunV3Error("final execution contract calibration policy differs from current policy")
        if contract["evaluator"] != policy["evaluator"]:
            raise RunV3Error("final execution contract evaluator differs from current policy")
        seal_ref = require_production_ref(target["seal_ref"], "final seal_ref")
        manifests = target.get("arm_manifests")
        manifest_refs = target.get("arm_manifest_refs")
        seal = target.get("seal")
        if (not isinstance(manifests, Mapping) or not isinstance(manifest_refs, Mapping) or
                not isinstance(seal, Mapping) or set(manifests) != set(contract["arms"]) or
                set(manifest_refs) != set(contract["arms"])):
            raise RunV3Error("sealed target must retain every exact arm manifest and ref")
        try:
            execution.validate_final_seal(seal, manifests)
            execution.validate_artifact_binding(seal_ref, seal, "final seal_ref")
        except execution.ContractError as exc:
            raise RunV3Error(str(exc)) from exc
        if seal["arm_manifest_refs"] != manifest_refs:
            raise RunV3Error("sealed target arm manifest refs differ from final seal")
        profile = _execution_profile_for(policy, contract["target"]["model_name"], "arm")
        runtime = {"package": stado.RUNTIME_PACKAGE,
                   "device": contract["runtime_evidence"]["device"],
                   "revision": policy["revisions"]["runtime"]}
        for arm in contract["arms"]:
            manifest = manifests[arm]
            manifest_ref = require_production_ref(
                manifest_refs[arm], f"final {arm} arm manifest_ref",
            )
            try:
                execution.validate_arm_manifest(manifest, contract)
                execution.validate_artifact_binding(
                    manifest_ref, manifest, f"final {arm} arm manifest_ref",
                )
            except execution.ContractError as exc:
                raise RunV3Error(str(exc)) from exc
            command = [
                *FINAL_TEST_WORKER_COMMAND,
                "--seal-ref", seal_ref["uri"],
                "--seal-ref-generation", seal_ref["generation"],
                "--arm-manifest", manifest_ref["uri"],
                "--arm-manifest-generation", manifest_ref["generation"],
                "--attempt-number", str(attempt),
                "--device", contract["runtime_evidence"]["device"],
            ]
            job = stado.build_job(
                phase="arm", command=command, policy_sha256=policy_bundle["policy_sha256"],
                code_commit=contract["revisions"]["code"],
                resources={key: profile[key] for key in ("accelerator", "memory_bytes", "runtime_seconds")},
                image=profile["image"], dependency_lock_ref=profile["dependency_lock_ref"],
                secrets=[], runtime=runtime,
                output_prefix=manifest["output_prefix"], input_refs=[seal_ref, manifest_ref],
                dependencies=dependencies, seal_ref=seal_ref, manifest_ref=manifest_ref,
                attempt=attempt,
                target_id=target_id, arm=arm, contract_sha256=contract["contract_sha256"],
            )
            jobs.append(job)
    return jobs


PREFLIGHT_PLAN_KEYS = frozenset({
    "schema_version", "plan_kind", "inventory_plan_ref", "inventory_plan_sha256",
    "selection_ref", "selection_sha256", "target_count", "targets", "plan_sha256",
})
PREFLIGHT_PLAN_TARGET_KEYS = frozenset({
    "target_id", "descriptor_sha256", "descriptor_ref", "output_prefix", "node_id",
})


def validate_preflight_plan(plan: Mapping[str, Any], *, production: bool = False) -> None:
    if not isinstance(plan, Mapping) or set(plan) != set(PREFLIGHT_PLAN_KEYS):
        raise RunV3Error("preflight submission plan keys are malformed")
    if plan["schema_version"] != SCHEMA_VERSION or plan["plan_kind"] != "desired-results-preflight-plan-v3":
        raise RunV3Error("preflight submission plan schema/kind differs")
    _artifact_ref(plan["inventory_plan_ref"], "preflight plan inventory_plan_ref")
    if not isinstance(plan["inventory_plan_sha256"], str) or len(plan["inventory_plan_sha256"]) != 64:
        raise RunV3Error("preflight plan inventory_plan_sha256 is invalid")
    selection_ref = plan["selection_ref"]
    selection_sha = plan["selection_sha256"]
    if (selection_ref is None) != (selection_sha is None):
        raise RunV3Error("preflight plan selection ref/hash must both be null or both be present")
    if selection_ref is not None:
        _artifact_ref(selection_ref, "preflight plan selection_ref")
        if not isinstance(selection_sha, str) or len(selection_sha) != 64:
            raise RunV3Error("preflight plan selection_sha256 is invalid")
    targets = plan["targets"]
    if not isinstance(targets, list) or not targets or plan["target_count"] != len(targets):
        raise RunV3Error("preflight plan target_count differs")
    seen: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping) or set(target) != set(PREFLIGHT_PLAN_TARGET_KEYS):
            raise RunV3Error(f"preflight plan target {index} keys are malformed")
        target_id = target["target_id"]
        if not isinstance(target_id, str) or not target_id or target_id in seen:
            raise RunV3Error("preflight plan target IDs must be unique non-empty strings")
        seen.add(target_id)
        _artifact_ref(target["descriptor_ref"], f"preflight plan target {index}.descriptor_ref")
        if not isinstance(target["descriptor_sha256"], str) or len(target["descriptor_sha256"]) != 64:
            raise RunV3Error("preflight plan descriptor_sha256 is invalid")
        expected_node = {key: target[key] for key in PREFLIGHT_PLAN_TARGET_KEYS if key != "node_id"}
        if target["node_id"] != execution.content_id("preflight-node-v3", expected_node):
            raise RunV3Error("preflight plan node_id differs from its immutable inputs")
        if not isinstance(target["output_prefix"], str) or not target["output_prefix"].startswith("gs://"):
            raise RunV3Error("preflight plan output_prefix must be a gs:// URI")
    if [target["target_id"] for target in targets] != sorted(seen):
        raise RunV3Error("preflight plan targets are not sorted by target_id")
    unhashed = dict(plan)
    supplied_sha = unhashed.pop("plan_sha256")
    if supplied_sha != canonical_sha256(unhashed):
        raise RunV3Error("preflight plan_sha256 mismatch")
    if production:
        require_production_document(plan, "preflight submission plan")


def build_preflight_plan(bindings: InventoryPlanBindings, *, output_namespace: str,
                         selection: Mapping[str, Any] | None = None,
                         selection_ref: Mapping[str, Any] | None = None,
                         full_default: bool = False) -> dict[str, Any]:
    """Build an acyclic submission for one explicit pilot or explicit full inventory."""
    if selection is None and not full_default:
        raise RunV3Error("preflight planning requires a selection or explicit full_default=True")
    if selection is not None and full_default:
        raise RunV3Error("selection and full_default are mutually exclusive")
    output_namespace = _production_output_namespace(output_namespace, "output_namespace")
    by_target = {binding.payload["target"]["target_id"]: binding for binding in bindings}
    if selection is None:
        if selection_ref is not None:
            raise RunV3Error("selection_ref is incompatible with full_default")
        selected_ids = sorted(by_target)
        effective_selection_ref = None
        selection_sha = None
    else:
        selected_ids, selection_sha = _validated_inventory_selection(selection, bindings.payload)
        if selection_ref is None:
            effective_selection_ref = _offline_ref("inventory-selections", selection)
        else:
            try:
                execution.validate_artifact_binding(selection_ref, selection, "selection_ref")
            except execution.ContractError as exc:
                raise RunV3Error(str(exc)) from exc
            effective_selection_ref = dict(selection_ref)
    if set(selected_ids) - set(by_target):
        raise RunV3Error("inventory selection contains an unknown target")
    targets: list[dict[str, Any]] = []
    for target_id in selected_ids:
        binding = by_target[target_id]
        descriptor = binding.payload
        digest = descriptor["descriptor_sha256"]
        target_key = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]
        node_payload = {
            "target_id": target_id,
            "descriptor_sha256": digest,
            "descriptor_ref": binding.ref,
            "output_prefix": f"{output_namespace.rstrip('/')}/preflight/{target_key}/{digest}",
        }
        node_payload["node_id"] = execution.content_id("preflight-node-v3", node_payload)
        targets.append(node_payload)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "plan_kind": "desired-results-preflight-plan-v3",
        "inventory_plan_ref": bindings.ref,
        "inventory_plan_sha256": bindings.inventory_plan_sha256,
        "selection_ref": effective_selection_ref,
        "selection_sha256": selection_sha,
        "target_count": len(targets),
        "targets": targets,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    validate_preflight_plan(payload)
    return payload


def publish_preflight_plan(store: Any, plan: Mapping[str, Any], *,
                           control_prefix: str, full_default: bool = False) -> dict[str, str]:
    """Publish one explicitly authorized canonical submission manifest."""
    if plan.get("selection_ref") is None and not full_default:
        raise RunV3Error("publishing a full inventory requires explicit full_default=True")
    if plan.get("selection_ref") is not None and full_default:
        raise RunV3Error("selection_ref and full_default are mutually exclusive")
    validate_preflight_plan(plan, production=True)
    inventory, inventory_ref = _load_inventory_plan_ref(
        store, plan["inventory_plan_ref"], production=True,
    )
    if (inventory_ref != execution.validate_artifact_ref(plan["inventory_plan_ref"]) or
            inventory.get("plan_sha256") != plan["inventory_plan_sha256"]):
        raise RunV3Error("preflight plan inventory lineage differs from exact store bytes")
    inventory_targets = {item["target_id"]: item for item in inventory["descriptors"]}
    for target in plan["targets"]:
        source = inventory_targets.get(target["target_id"])
        if (source is None or source.get("descriptor_sha256") != target["descriptor_sha256"] or
                source.get("descriptor_ref") != target["descriptor_ref"]):
            raise RunV3Error("preflight plan target differs from exact inventory lineage")
        _verify_production_ref(store, target["descriptor_ref"],
                               f"preflight descriptor {target['target_id']}")
    submitted_target_ids = [item["target_id"] for item in plan["targets"]]
    if plan["selection_ref"] is not None:
        selection, selection_ref = load_inventory_selection(
            store, plan["selection_ref"], inventory, production=True,
        )
        if (selection_ref != execution.validate_artifact_ref(plan["selection_ref"])
                or selection["selection_sha256"] != plan["selection_sha256"]
                or selection["target_ids"] != submitted_target_ids):
            raise RunV3Error("preflight plan selection differs from exact store bytes")
    elif submitted_target_ids != sorted(inventory_targets):
        raise RunV3Error("full-default preflight plan does not contain the exact inventory")
    prefix = _production_output_namespace(control_prefix, "control_prefix").rstrip("/")
    uri = f"{prefix}/preflight/submissions/{plan['plan_sha256']}.json"
    return publish_json(store, uri, plan)


def load_preflight_plan(store: Any, submission_ref: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    plan, normalized = _production_document(store, submission_ref, "preflight submission plan")
    validate_preflight_plan(plan, production=True)
    expected_suffix = f"/preflight/submissions/{plan['plan_sha256']}.json"
    if not normalized["uri"].endswith(expected_suffix):
        raise RunV3Error("submission_ref is outside its canonical content-addressed path")
    return plan, normalized


def load_production_inventory_bindings(
    store: Any, inventory_plan_ref: Mapping[str, Any],
) -> InventoryPlanBindings:
    """Materialize production planning bindings only from exact published store bytes."""
    plan, normalized_plan_ref = _load_inventory_plan_ref(
        store, inventory_plan_ref, production=True,
    )
    plan_sha = plan.get("plan_sha256")
    if not isinstance(plan_sha, str) or len(plan_sha) != 64:
        raise RunV3Error("published inventory plan lacks a canonical plan_sha256")
    bindings: list[InventoryBinding] = []
    for index, entry in enumerate(plan["descriptors"]):
        descriptor_ref = entry.get("descriptor_ref")
        if descriptor_ref is None:
            raise RunV3Error(f"published inventory descriptor {index} lacks descriptor_ref")
        descriptor, normalized_descriptor_ref = load_descriptor(
            store, descriptor_ref, expected_sha256=entry["descriptor_sha256"], production=True,
        )
        if descriptor["target"]["target_id"] != entry["target_id"]:
            raise RunV3Error("published inventory descriptor target differs from its exact plan entry")
        bindings.append(InventoryBinding(descriptor, normalized_descriptor_ref))
    return InventoryPlanBindings(plan, normalized_plan_ref, bindings, plan_sha)


def publish_preflight_submission(
    store: Any, inventory_plan_ref: Mapping[str, Any], *, output_namespace: str,
    control_prefix: str, selection_ref: Mapping[str, Any] | None = None,
    full_default: bool = False,
) -> dict[str, Any]:
    """Load published lineage and publish one explicitly selected submission exactly once."""
    if selection_ref is None and not full_default:
        raise RunV3Error("preflight submission requires selection_ref or explicit full_default=True")
    if selection_ref is not None and full_default:
        raise RunV3Error("selection_ref and full_default are mutually exclusive")
    bindings = load_production_inventory_bindings(store, inventory_plan_ref)
    if selection_ref is None:
        selection = None
        normalized_selection_ref = None
    else:
        selection, normalized_selection_ref = load_inventory_selection(
            store, selection_ref, bindings.payload, production=True,
        )
    plan = build_preflight_plan(
        bindings, output_namespace=output_namespace, selection=selection,
        selection_ref=normalized_selection_ref, full_default=full_default,
    )
    submission_ref = publish_preflight_plan(
        store, plan, control_prefix=control_prefix, full_default=full_default,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "submission": plan,
        "submission_ref": submission_ref,
        "submission_sha256": plan["plan_sha256"],
    }


def _read_json_file(path: str | os.PathLike[str] | Path, label: str) -> Any:
    raw = Path(path).read_bytes()
    return _strict_json(raw, label, allow_trailing_newline=True)


def _optional_ref(value: str | None, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    candidate: Any
    if value.lstrip().startswith("{"):
        try:
            candidate = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RunV3Error(f"{label} is invalid JSON: {exc}") from exc
    else:
        candidate = _read_json_file(value, label)
    return _artifact_ref(candidate, label)


def _write_cli_result(path: Path | None, value: Any) -> None:
    if path is not None:
        atomic_write_json(path, value)
    print(canonical_bytes(value).decode("ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="desired-results-run-v3")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    plan_parser = subparsers.add_parser("plan", help="build an offline immutable plan")
    source = plan_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy-bundle", type=Path)
    source.add_argument("--inventory-plan", type=Path)
    plan_parser.add_argument("--descriptor-dir", type=Path)
    selection_mode = plan_parser.add_mutually_exclusive_group()
    selection_mode.add_argument("--selection", type=Path)
    selection_mode.add_argument("--full-default", action="store_true")
    plan_parser.add_argument("--selection-ref")
    plan_parser.add_argument("--policy-ref")
    plan_parser.add_argument("--target-id", action="append")
    plan_parser.add_argument("--receipt-history", type=Path)
    plan_parser.add_argument("--output-namespace")
    plan_parser.add_argument("--output", type=Path)

    preflight_submit_parser = subparsers.add_parser(
        "preflight-submit", help="publish an exact production preflight submission manifest",
    )
    preflight_submit_parser.add_argument("--inventory-plan-ref", required=True)
    submit_selection_mode = preflight_submit_parser.add_mutually_exclusive_group(required=True)
    submit_selection_mode.add_argument("--selection-ref")
    submit_selection_mode.add_argument("--full-default", action="store_true")
    preflight_submit_parser.add_argument("--output-namespace", required=True)
    preflight_submit_parser.add_argument("--control-prefix", required=True)
    preflight_submit_parser.add_argument("--output", type=Path)
    preflight_submit_parser.add_argument(
        "--execute", action="store_true", help="exact-load lineage and create the submission manifest",
    )

    submit_parser = subparsers.add_parser("submit", help="publish manifests and optionally dispatch calibration jobs")
    submit_parser.add_argument("--policy-bundle", type=Path, required=True)
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--control-prefix", required=True)
    submit_parser.add_argument("--output", type=Path)
    submit_parser.add_argument("--execute", action="store_true", help="perform create-only writes and Stado submission")

    promote_parser = subparsers.add_parser("promote", help="promote complete preflight receipts")
    promote_parser.add_argument("--receipt", type=Path, action="append", required=True)
    promote_parser.add_argument("--output", type=Path)
    promote_parser.add_argument("--execute", action="store_true", help="enforce production-only immutable refs")

    seal_parser = subparsers.add_parser("seal", help="seal calibrated contracts, arms, and final waves")
    seal_parser.add_argument("--policy-bundle", type=Path, required=True)
    seal_parser.add_argument("--calibration-receipts", type=Path, required=True)
    seal_parser.add_argument("--policy-ref")
    seal_parser.add_argument("--evaluator-ref")
    seal_parser.add_argument("--final-attempt", type=int, default=1)
    seal_parser.add_argument("--control-prefix", required=True)
    seal_parser.add_argument("--output", type=Path)
    seal_parser.add_argument("--execute", action="store_true", help="perform create-only production sealing")

    finalize_parser = subparsers.add_parser("finalize", help="publish final result after exact arm completion")
    finalize_parser.add_argument("--sealed-target", type=Path, required=True)
    finalize_parser.add_argument("--completion", type=Path, action="append", required=True)
    finalize_parser.add_argument("--control-prefix", required=True)
    finalize_parser.add_argument("--output", type=Path)
    finalize_parser.add_argument("--execute", action="store_true", help="perform create-only final publication")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.phase == "plan":
            if args.policy_bundle is not None:
                if args.selection is not None or args.selection_ref is not None or args.full_default:
                    raise RunV3Error("inventory selection options require --inventory-plan")
                bundle = _read_json_file(args.policy_bundle, "policy bundle")
                receipt_history = (
                    _read_json_file(args.receipt_history, "calibration receipt history")
                    if args.receipt_history is not None else None
                )
                value = plan(
                    bundle, policy_ref=_optional_ref(args.policy_ref, "policy_ref"),
                    target_ids=args.target_id, receipt_history=receipt_history,
                )
            else:
                if args.receipt_history is not None:
                    raise RunV3Error("--receipt-history requires --policy-bundle")
                if args.descriptor_dir is None or args.output_namespace is None:
                    raise RunV3Error("inventory planning requires --descriptor-dir and --output-namespace")
                bindings = load_inventory_plan(args.inventory_plan, args.descriptor_dir)
                selection = _read_json_file(args.selection, "inventory selection") if args.selection else None
                value = build_preflight_plan(
                    bindings, output_namespace=args.output_namespace, selection=selection,
                    selection_ref=_optional_ref(args.selection_ref, "selection_ref"),
                    full_default=args.full_default,
                )
        elif args.phase == "preflight-submit":
            inventory_ref = _optional_ref(args.inventory_plan_ref, "inventory_plan_ref")
            selection_ref = _optional_ref(args.selection_ref, "selection_ref")
            if not args.execute:
                value = {
                    "execute": False, "inventory_plan_ref": inventory_ref,
                    "selection_ref": selection_ref, "full_default": args.full_default,
                    "output_namespace": args.output_namespace,
                    "control_prefix": args.control_prefix,
                }
            else:
                value = publish_preflight_submission(
                    GCSStore(), inventory_ref, output_namespace=args.output_namespace,
                    control_prefix=args.control_prefix, selection_ref=selection_ref,
                    full_default=args.full_default,
                )
                value["execute"] = True
        elif args.phase == "submit":
            bundle = _read_json_file(args.policy_bundle, "policy bundle")
            run_plan = _read_json_file(args.plan, "calibration plan")
            value = submit(GCSStore() if args.execute else LocalStore(), bundle, run_plan,
                           control_prefix=args.control_prefix, execute=args.execute)
        elif args.phase == "promote":
            receipts = [_read_json_file(path, f"preflight receipt {index}")
                        for index, path in enumerate(args.receipt)]
            value = promote(receipts, store=GCSStore() if args.execute else None,
                            production=args.execute)
        elif args.phase == "seal":
            bundle = _read_json_file(args.policy_bundle, "policy bundle")
            receipts = _read_json_file(args.calibration_receipts, "calibration receipts")
            value = seal(GCSStore() if args.execute else LocalStore(), bundle, receipts,
                         control_prefix=args.control_prefix, execute=args.execute,
                         policy_ref=_optional_ref(args.policy_ref, "policy_ref"),
                         evaluator_ref=_optional_ref(args.evaluator_ref, "evaluator_ref"),
                         final_attempt=args.final_attempt)
        else:
            sealed_target = _read_json_file(args.sealed_target, "sealed target")
            completions = [_read_json_file(path, f"completion {index}")
                           for index, path in enumerate(args.completion)]
            value = finalize(GCSStore() if args.execute else LocalStore(), sealed_target, completions,
                             control_prefix=args.control_prefix, execute=args.execute)
        _write_cli_result(args.output, value)
        return 0
    except (RunV3Error, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"desired-results-run-v3 failed: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
