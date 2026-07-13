#!/usr/bin/env python3
"""Build the immutable desired-results v3 experiment policy bundle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

try:
    from . import desired_results_stado, desired_results_target
    from .desired_results_execution_contract import (
        ContractError,
        artifact_ref,
        canonical_json,
        canonical_sha256,
        validate_artifact_binding,
        validate_artifact_ref,
    )
except (ImportError, ModuleNotFoundError):
    _CONTRACT_PATH = Path(__file__).with_name("desired_results_execution_contract.py")
    _CONTRACT_SPEC = importlib.util.spec_from_file_location("desired_results_execution_contract_v3", _CONTRACT_PATH)
    if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
        raise ImportError(f"cannot load execution contract from {_CONTRACT_PATH}")
    _contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
    _CONTRACT_SPEC.loader.exec_module(_contract)
    desired_results_target = _contract.desired_results_target
    _STADO_PATH = Path(__file__).with_name("desired_results_stado.py")
    _STADO_SPEC = importlib.util.spec_from_file_location("desired_results_stado_v3", _STADO_PATH)
    if _STADO_SPEC is None or _STADO_SPEC.loader is None:
        raise ImportError(f"cannot load Stado contract from {_STADO_PATH}")
    desired_results_stado = importlib.util.module_from_spec(_STADO_SPEC)
    _STADO_SPEC.loader.exec_module(desired_results_stado)
    ContractError = _contract.ContractError
    artifact_ref = _contract.artifact_ref
    canonical_json = _contract.canonical_json
    canonical_sha256 = _contract.canonical_sha256
    validate_artifact_binding = _contract.validate_artifact_binding
    validate_artifact_ref = _contract.validate_artifact_ref

SCHEMA_VERSION = 3
BUNDLE_KIND = "desired-results-policy-bundle-v3"
POLICY_KIND = "desired-results-run-policy-v3"
BASELINE_KIND = "desired-results-baseline-config-v3"
METHODS = tuple(desired_results_target.METHODS)
STRATEGIES = tuple(desired_results_target.STRATEGIES)
PHASES = ("calibration", "seal", "arm", "finalize")
GROM_MAX_DIRECTIONS = 16
GROM_MAX_OPTIMIZATION_STEPS = 300
GROM_MAX_WARMUP_STEPS = 100

GROM_RUNTIME_CAPS = {
    "num_directions": GROM_MAX_DIRECTIONS,
    "optimization_steps": GROM_MAX_OPTIMIZATION_STEPS,
    "warmup_steps": GROM_MAX_WARMUP_STEPS,
}
GROM_POSITIVE_FLOAT_BOUNDS = {
    "behavior_weight": (1e-3, 10.0),
    "caa_alignment_weight": (1e-3, 10.0),
    "concentration_weight": (1e-3, 10.0),
    "contrastive_weight": (1e-3, 10.0),
    "create_noise_scale": (1e-4, 1.0),
    "gate_temperature": (0.1, 10.0),
    "gate_warmup_weight": (1e-3, 10.0),
    "independence_weight": (1e-3, 10.0),
    "learning_rate": (1e-5, 1e-2),
    "max_alpha": (0.1, 10.0),
    "max_grad_norm": (0.1, 10.0),
    "smooth_weight": (1e-3, 10.0),
    "sparse_weight": (1e-3, 10.0),
    "strength": (1e-3, 10.0),
    "utility_weight": (1e-3, 10.0),
    "weight_decay": (1e-6, 1e-1),
}
GROM_DIMENSION_TRIPLES = (
    ("gate_dim_min", "gate_hidden_dim", "gate_dim_max"),
    ("intensity_dim_min", "intensity_hidden_dim", "intensity_dim_max"),
)
DEFAULT_CALIBRATION_POLICY = {
    "name": "optuna",
    "version": "1",
    "options": {
        "device": "cuda",
        "optimizer": {"backend": "optuna", "direction": "maximize", "seed": 0},
    },
}

PolicyError = ContractError


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ContractError(f"{label} keys must be exactly {sorted(keys)!r}; got {actual!r}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value



def validate_production_output_namespace(value: Any, label: str = "output_namespace") -> str:
    """Require one normalized GCS bucket plus a non-empty object prefix."""
    namespace = _string(value, label)
    parsed = urlsplit(namespace)
    if f"{parsed.scheme}://{parsed.netloc}{parsed.path}" != namespace:
        raise PolicyError(f"{label} must not contain a query or fragment")
    if (parsed.scheme, bool(parsed.netloc), parsed.path.startswith("/")) != ("gs", True, True):
        raise PolicyError(f"{label} must be a canonical gs://bucket/prefix namespace")
    if parsed.geturl() != namespace or urlsplit(namespace.replace("\\", "/")) != parsed:
        raise PolicyError(f"{label} must be a canonical gs://bucket/prefix namespace")
    bucket = parsed.netloc
    prefix = parsed.path.removeprefix("/")
    if (not prefix or namespace.endswith("/") or
            any(character.isspace() for character in namespace)):
        raise PolicyError(f"{label} must be a canonical gs://bucket/prefix namespace")
    if (bucket != bucket.lower() or not bucket.replace("-", "").replace("_", "").replace(".", "").isalnum() or
            not bucket[0].isalnum() or not bucket[-1].isalnum()):
        raise PolicyError(f"{label} has an invalid GCS bucket name")
    segments = prefix.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise PolicyError(f"{label} must not contain empty or dot path segments")
    return namespace

def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    canonical_json(value)
    return copy.deepcopy(dict(value))


def _serialize_parameter(param: Any) -> dict[str, Any]:
    """Serialize a canonical Wisent Param without lying about its distribution."""
    from wisent.core.utils.services.optimization.core.parameters import (
        CategoricalParam,
        FloatParam,
        IntParam,
    )

    if isinstance(param, CategoricalParam):
        return {"kind": "categorical", "choices": copy.deepcopy(list(param.choices))}
    raw = asdict(param)
    if isinstance(param, FloatParam):
        return {
            "kind": "float",
            "distribution": raw["distribution"],
            "mu": raw["mu"],
            "sigma": raw["sigma"],
            "low": raw["low"],
            "high": raw["high"],
            "log_scale": raw["log_scale"],
        }
    if isinstance(param, IntParam):
        return {
            "kind": "int",
            "distribution": raw["distribution"],
            "mu": raw["mu"],
            "sigma": raw["sigma"],
            "q": raw["q"],
            "low": raw["low"],
            "high": raw["high"],
        }
    raise ContractError(f"unsupported canonical parameter {type(param).__name__}")


def _cap_integer(spec: dict[str, Any], maximum: int, label: str) -> None:
    maximum = _positive(maximum, f"{label} cap")
    if spec["kind"] == "categorical":
        choices = [choice for choice in spec["choices"] if type(choice) is int and 0 <= choice <= maximum]
        if not choices:
            raise ContractError(f"{label} has no value within the model cap {maximum}")
        spec["choices"] = choices
        return
    if spec["kind"] != "int":
        raise ContractError(f"{label} is not an integer parameter")
    # Integer high is exclusive everywhere in Wisent's optimization contract.
    old_high = spec["high"]
    spec["high"] = maximum + 1 if old_high is None else min(old_high, maximum + 1)
    if spec["low"] is None:
        spec["low"] = max(1, spec["q"]) if spec["distribution"] == "qlognormal" else 0
    if spec["low"] >= spec["high"]:
        raise ContractError(f"{label} has empty support after applying cap {maximum}")


def _bound_positive_float(
    spec: dict[str, Any], low: float, high: float, label: str,
) -> None:
    if spec.get("kind") != "float" or not 0.0 < low < high:
        raise ContractError(f"{label} cannot be bounded to a positive finite range")
    spec.clear()
    spec.update({
        "kind": "float", "distribution": "uniform",
        "mu": None, "sigma": None, "low": low, "high": high,
        "log_scale": True,
    })


def _make_grom_dimension_triple_safe(
    space: dict[str, Any], names: tuple[str, str, str], hidden_size: int,
) -> None:
    minimum_name, hidden_name, maximum_name = names
    minimum = space[minimum_name].get("low")
    maximum_exclusive = space[maximum_name].get("high")
    if type(minimum) is not int or type(maximum_exclusive) is not int:
        raise ContractError(f"grom dimension triple {names!r} lacks finite integer bounds")
    maximum = max(minimum, min(hidden_size, maximum_exclusive - 1))
    space[minimum_name] = {"kind": "categorical", "choices": [minimum]}
    space[hidden_name] = {
        "kind": "categorical", "choices": list(range(minimum, maximum + 1)),
    }
    space[maximum_name] = {"kind": "categorical", "choices": [maximum]}


def get_effective_method_space(method: str, layer_count: int, hidden_size: int) -> dict[str, Any]:
    """Return the exact canonical method space with target-specific finite caps."""
    if method not in METHODS:
        raise ContractError(f"unknown method {method!r}")
    layer_count = _positive(layer_count, "layer_count")
    hidden_size = _positive(hidden_size, "hidden_size")
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.search_space import get_method_space

    # The upstream API receives the model layer count including its final, unsafe
    # output block.  TargetManifest layer_count is the eligible 1-based route count.
    raw = get_method_space(method, layer_count + 1)
    space = {name: _serialize_parameter(param) for name, param in sorted(raw.items())}
    for name in ("layer", "sensor_layer", "steering_start", "steering_end"):
        if name in space:
            space[name] = {"kind": "categorical", "choices": list(range(1, layer_count + 1))}

    hidden_caps = {
        "mlp": ("hidden_dim",),
        "nurt": ("num_dims", "max_concept_dim", "flow_hidden_dim"),
        "grom": (
            "num_directions", "gate_hidden_dim", "intensity_hidden_dim",
            "gate_dim_min", "gate_dim_max", "intensity_dim_min", "intensity_dim_max",
        ),
        "tecza": ("num_directions", "max_directions"),
        "wicher": ("concept_dim",),
    }
    for name in hidden_caps.get(method, ()):
        if name in space:
            _cap_integer(space[name], hidden_size, f"{method}.{name}")
    if method == "grom":
        for name, maximum in GROM_RUNTIME_CAPS.items():
            if name in space:
                cap = min(hidden_size, maximum) if name == "num_directions" else maximum
                _cap_integer(space[name], cap, f"{method}.{name}")
        for names in GROM_DIMENSION_TRIPLES:
            _make_grom_dimension_triple_safe(space, names, hidden_size)
        for name, bounds in GROM_POSITIVE_FLOAT_BOUNDS.items():
            if name in space:
                _bound_positive_float(space[name], *bounds, f"{method}.{name}")

    canonical_json(space)
    return space


def _validate_resource_class(value: Any, label: str) -> dict[str, Any]:
    item = _exact(
        value,
        {"accelerator", "memory_bytes", "runtime_seconds", "image", "dependency_lock_ref"},
        label,
    )
    image = _exact(item["image"], {"name", "project"}, f"{label}.image")
    normalized_image = {
        "name": _string(image["name"], f"{label}.image.name"),
        "project": _string(image["project"], f"{label}.image.project"),
    }
    installed_image = {
        "name": desired_results_stado.STADO_IMAGE_NAME,
        "project": desired_results_stado.STADO_IMAGE_PROJECT,
    }
    if normalized_image != installed_image:
        raise ContractError(f"{label}.image must equal the installed immutable Stado image identity")
    return {
        "accelerator": _string(item["accelerator"], f"{label}.accelerator"),
        "memory_bytes": _positive(item["memory_bytes"], f"{label}.memory_bytes"),
        "runtime_seconds": _positive(item["runtime_seconds"], f"{label}.runtime_seconds"),
        "image": normalized_image,
        "dependency_lock_ref": validate_artifact_ref(
            item["dependency_lock_ref"], f"{label}.dependency_lock_ref"
        ),
    }


def _validate_resource_classes(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError("resource_classes must be a non-empty object")
    result: dict[str, Any] = {}
    for name, raw in sorted(value.items()):
        _string(name, "resource class name")
        result[name] = _validate_resource_class(raw, f"resource_classes.{name}")
    return result




def _validate_model_classes(value: Any, resources: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError("model_classes must be a non-empty object")
    result: dict[str, Any] = {}
    for model_name, raw in sorted(value.items()):
        _string(model_name, "model_classes key")
        item = _exact(raw, {"phase_classes", "hidden_size"}, f"model_classes.{model_name}")
        phases = _exact(item["phase_classes"], set(PHASES), f"model_classes.{model_name}.phase_classes")
        normalized_phases: dict[str, str] = {}
        for phase in PHASES:
            class_name = _string(phases[phase], f"model_classes.{model_name}.phase_classes.{phase}")
            if class_name not in resources:
                raise ContractError(f"unknown resource class {class_name!r} for {model_name}/{phase}")
            normalized_phases[phase] = class_name
        result[model_name] = {
            "phase_classes": normalized_phases,
            "hidden_size": _positive(item["hidden_size"], f"model_classes.{model_name}.hidden_size"),
        }
    return result


def resource_for(policy: Mapping[str, Any], model_name: str, phase: str) -> dict[str, Any]:
    """Return the exact immutable resource profile selected for a model phase."""
    model_name = _string(model_name, "model_name")
    if phase not in PHASES:
        raise ContractError(f"unknown execution phase {phase!r}")
    if not isinstance(policy, Mapping):
        raise ContractError("policy must be an object")
    model_classes = policy.get("model_classes")
    resource_classes = policy.get("resource_classes")
    if not isinstance(model_classes, Mapping) or not isinstance(resource_classes, Mapping):
        raise ContractError("policy must contain model_classes and resource_classes objects")
    if model_name not in model_classes:
        raise ContractError(f"model_classes is missing exact model {model_name!r}")
    model = _exact(
        model_classes[model_name], {"phase_classes", "hidden_size"},
        f"model_classes.{model_name}",
    )
    phases = _exact(
        model["phase_classes"], set(PHASES), f"model_classes.{model_name}.phase_classes"
    )
    class_name = _string(phases[phase], f"model_classes.{model_name}.phase_classes.{phase}")
    if class_name not in resource_classes:
        raise ContractError(f"unknown resource class {class_name!r} for {model_name}/{phase}")
    return _validate_resource_class(resource_classes[class_name], f"resource_classes.{class_name}")


def _validate_evaluator(value: Any) -> dict[str, Any]:
    root = _exact(value, {"name", "version", "options"}, "evaluator")
    return {
        "name": _string(root["name"], "evaluator.name"),
        "version": _string(root["version"], "evaluator.version"),
        "options": _json_mapping(root["options"], "evaluator.options"),
    }


def _validate_calibration_policy(value: Any) -> dict[str, Any]:
    """Validate the policy-level template used to derive sealed job controls."""
    root = _exact(value, {"name", "version", "options"}, "calibration_policy")
    options = _exact(root["options"], {"device", "optimizer"}, "calibration_policy.options")
    optimizer = _exact(
        options["optimizer"], {"backend", "direction", "seed"},
        "calibration_policy.options.optimizer",
    )
    backend = _string(optimizer["backend"], "calibration_policy.options.optimizer.backend")
    direction = _string(optimizer["direction"], "calibration_policy.options.optimizer.direction")
    if direction not in {"maximize", "minimize"}:
        raise ContractError("calibration_policy optimizer direction must be maximize or minimize")
    seed = optimizer["seed"]
    if type(seed) is not int or seed < 0:
        raise ContractError("calibration_policy optimizer seed must be a non-negative integer")
    return {
        "name": _string(root["name"], "calibration_policy.name"),
        "version": _string(root["version"], "calibration_policy.version"),
        "options": {
            "device": _string(options["device"], "calibration_policy.options.device"),
            "optimizer": {"backend": backend, "direction": direction, "seed": seed},
        },
    }
def calibration_policy_for(
    policy: Mapping[str, Any],
    policy_ref: Mapping[str, Any],
    target_id: str,
    method: str,
) -> dict[str, Any]:
    """Derive exact self-contained CalibrationManifest optimizer controls."""
    target_id = _string(target_id, "target_id")
    if method not in METHODS:
        raise ContractError(f"unknown method {method!r}")
    validate_artifact_binding(policy_ref, policy, "calibration_policy.policy_ref")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("method_spaces"), Mapping):
        raise ContractError("policy method_spaces are malformed")
    if target_id not in policy["method_spaces"]:
        raise ContractError(f"policy must contain exactly one target {target_id!r}")
    base = _validate_calibration_policy(policy.get("calibration_policy"))
    method_space = copy.deepcopy(policy["method_spaces"][target_id].get(method))
    if not isinstance(method_space, Mapping) or not method_space:
        raise ContractError(f"target {target_id} lacks method space {method}")
    base_optimizer = base["options"]["optimizer"]
    optimizer = {
        "backend": base_optimizer["backend"],
        "direction": base_optimizer["direction"],
        "seed": base_optimizer["seed"],
        "trials_per_strategy": _positive(
            policy.get("trials_per_strategy", {}).get(method), f"trials_per_strategy.{method}",
        ),
        "method_space": method_space,
    }
    return {
        "name": base["name"],
        "version": base["version"],
        "policy_ref": validate_artifact_ref(policy_ref, "calibration_policy.policy_ref"),
        "options": {"device": base["options"]["device"], "optimizer": optimizer},
    }


def _baseline_binding(target_id: str, baseline_name: str) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_kind": BASELINE_KIND,
        "target_id": target_id,
        "arm_name": baseline_name,
        "method": None,
        "parameters": {},
    }
    data = canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    ref = artifact_ref(f"bundle:///objects/{digest}.json", "local", str(len(data)), digest)
    return {"ref": ref, "payload": payload}


def get_method_space(method: str, layer_count: int, hidden_size: int) -> dict[str, Any]:
    """Compatibility-free public spelling for the effective canonical space."""
    return get_effective_method_space(method, layer_count, hidden_size)




def build_policy_bundle(
    target_manifest_refs: Sequence[Mapping[str, Any]],
    pair_text_refs: Mapping[str, Mapping[str, Any]],
    revisions: Mapping[str, Any],
    resource_classes: Mapping[str, Any],
    model_classes: Mapping[str, Any],
    output_namespace: str,
    trials_per_strategy: int | Mapping[str, int],
    retry_policy: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    *,
    calibration_policy: Mapping[str, Any] | None = None,
    baseline_name: str = "unsteered",
) -> dict[str, Any]:
    """Build one deterministic, byte-sealed multi-target policy and local objects."""
    if not isinstance(target_manifest_refs, Sequence) or isinstance(target_manifest_refs, (str, bytes)) or not target_manifest_refs:
        raise ContractError("target_manifest_refs must be a non-empty sequence")
    revisions_root = _exact(revisions, {"code", "runtime"}, "revisions")
    normalized_revisions = {
        "code": _string(revisions_root["code"], "revisions.code"),
        "runtime": _string(revisions_root["runtime"], "revisions.runtime"),
    }
    if normalized_revisions["runtime"] != normalized_revisions["code"]:
        raise ContractError("revisions.runtime must equal revisions.code for detached-checkout execution")
    resources = _validate_resource_classes(resource_classes)
    models = _validate_model_classes(model_classes, resources)
    output_namespace = _string(output_namespace, "output_namespace")
    baseline_name = _string(baseline_name, "baseline_name")
    if baseline_name in METHODS:
        raise ContractError("baseline_name must be disjoint from method names")
    evaluator_value = _validate_evaluator(evaluator)
    calibration_value = _validate_calibration_policy(calibration_policy or DEFAULT_CALIBRATION_POLICY)
    retry = _exact(retry_policy, {"calibration_max_attempts", "max_pre_test_attempts"}, "retry_policy")
    retry_value = {
        "calibration_max_attempts": _positive(retry["calibration_max_attempts"], "retry_policy.calibration_max_attempts"),
        "max_pre_test_attempts": _positive(retry["max_pre_test_attempts"], "retry_policy.max_pre_test_attempts"),
    }
    if retry_value["calibration_max_attempts"] > 3 or retry_value["max_pre_test_attempts"] > 3:
        raise ContractError("retry limits cannot exceed the shared execution contract maximum of 3")
    if type(trials_per_strategy) is int:
        trial_map = {method: _positive(trials_per_strategy, "trials_per_strategy") for method in METHODS}
    else:
        trial_root = _exact(trials_per_strategy, set(METHODS), "trials_per_strategy")
        trial_map = {method: _positive(trial_root[method], f"trials_per_strategy.{method}") for method in METHODS}

    if not isinstance(pair_text_refs, Mapping):
        raise ContractError("pair_text_refs must be a target-id object")
    target_bindings: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_binding in enumerate(target_manifest_refs):
        binding = _exact(raw_binding, {"ref", "payload"}, f"target_manifest_refs[{index}]")
        manifest = binding["payload"]
        desired_results_target.validate_target_manifest(manifest)
        ref = validate_artifact_binding(binding["ref"], manifest, f"target_manifest_refs[{index}].ref")
        target = manifest["target"]
        target_id = target["target_id"]
        if target_id in seen:
            raise ContractError(f"duplicate target manifest for {target_id}")
        seen.add(target_id)
        lifecycle = manifest["execution"]
        if (
            lifecycle["state"] != "unprepared"
            or lifecycle["blocked"] is not False
            or lifecycle["rerun_locked"] is not False
            or manifest["activation"]["eligible"] is not True
            or manifest["support"]["state"] != "prepared"
            or manifest["evaluation"]["split"] != "test"
        ):
            raise ContractError(
                f"target {target_id} is not activation-eligible for calibration/final planning"
            )
        layer_count = manifest["activation"]["layer_count"]
        if type(layer_count) is not int or layer_count < 1:
            raise ContractError(f"target {target_id} lacks a positive activation layer_count")
        source_revisions = _exact(
            manifest["revisions"],
            {"inventory_sha256", "model_revision", "tokenizer_revision", "activation_revision"},
            f"target {target_id} revisions",
        )
        for key in ("model_revision", "tokenizer_revision", "activation_revision"):
            _string(source_revisions[key], f"target {target_id} {key}")
        model_name = target["model_name"]
        if model_name not in models:
            raise ContractError(f"model_classes is missing exact model {model_name!r}")
        pair_ref = validate_artifact_ref(pair_text_refs.get(target_id), f"pair_text_refs.{target_id}")
        manifest_pair_ref = validate_artifact_ref(manifest["support"]["pair_texts_ref"], f"target {target_id} support.pair_texts_ref")
        if pair_ref != manifest_pair_ref:
            raise ContractError(f"pair_text_refs.{target_id} differs from promoted TargetManifestV2")
        hidden_size = models[model_name]["hidden_size"]
        spaces = {method: get_effective_method_space(method, layer_count, hidden_size) for method in METHODS}
        target_bindings.append({"ref": ref, "payload": copy.deepcopy(manifest)})
        targets.append({
            "target_id": target_id,
            "model_name": model_name,
            "model_slug": target["model_slug"],
            "benchmark": target["benchmark"],
            "target_manifest_ref": ref,
            "layer_count": layer_count,
            "hidden_size": hidden_size,
            "revisions": {
                "model": source_revisions["model_revision"],
                "tokenizer": source_revisions["tokenizer_revision"],
                "activation": source_revisions["activation_revision"],
            },
            "method_spaces": spaces,
        })
    if set(pair_text_refs) != seen:
        raise ContractError("pair_text_refs must have exact target coverage")
    target_bindings.sort(key=lambda item: item["payload"]["target"]["target_id"])
    targets.sort(key=lambda item: item["target_id"])
    normalized_pairs = {target_id: validate_artifact_ref(pair_text_refs[target_id], f"pair_text_refs.{target_id}") for target_id in sorted(seen)}

    objects = [_baseline_binding(target["target_id"], baseline_name) for target in targets]
    baselines = {target["target_id"]: objects[index]["ref"] for index, target in enumerate(targets)}
    policy_target_refs = {target["target_id"]: target["target_manifest_ref"] for target in targets}
    method_spaces = {target["target_id"]: target["method_spaces"] for target in targets}
    matrix: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        job_revisions = {
            "model": target["revisions"]["model"],
            "tokenizer": target["revisions"]["tokenizer"],
            "activation": target["revisions"]["activation"],
            "code": normalized_revisions["code"],
            "runtime": normalized_revisions["runtime"],
        }
        matrix[target["target_id"]] = [
            {
                "target_id": target["target_id"],
                "model_name": target["model_name"],
                "model_slug": target["model_slug"],
                "benchmark": target["benchmark"],
                "target_manifest_ref": target["target_manifest_ref"],
                "method": method,
                "strategy": strategy,
                "trials": trial_map[method],
                "revisions": job_revisions,
                "resource_class": models[target["model_name"]]["phase_classes"]["calibration"],
            }
            for method in METHODS for strategy in STRATEGIES
        ]
    policy = {
        "schema_version": SCHEMA_VERSION,
        "policy_kind": POLICY_KIND,
        "target_manifest_refs": policy_target_refs,
        "revisions": normalized_revisions,
        "resource_classes": resources,
        "model_classes": models,
        "output_namespace": output_namespace,
        "trials_per_strategy": trial_map,
        "retry_policy": retry_value,
        "evaluator": evaluator_value,
        "calibration_policy": calibration_value,
        "method_spaces": method_spaces,
        "matrix": matrix,
        "baselines": baselines,
    }
    policy_sha256 = canonical_sha256(policy)
    bundle_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "bundle_kind": BUNDLE_KIND,
        "target_manifest_refs": target_bindings,
        "pair_text_refs": normalized_pairs,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "objects": objects,
    }
    bundle = dict(bundle_without_hash)
    bundle["bundle_sha256"] = canonical_sha256(bundle_without_hash)
    return bundle


def validate_policy_bundle(
    bundle: Mapping[str, Any], *, allow_local_baselines: bool = True, production: bool = False,
) -> None:
    root = _exact(bundle, {
        "schema_version", "bundle_kind", "target_manifest_refs", "pair_text_refs",
        "policy", "policy_sha256", "objects", "bundle_sha256",
    }, "policy bundle")
    if root["schema_version"] != SCHEMA_VERSION or root["bundle_kind"] != BUNDLE_KIND:
        raise PolicyError("policy bundle schema/kind mismatch")
    if root["policy_sha256"] != canonical_sha256(root["policy"]):
        raise PolicyError("policy_sha256 mismatch")
    unhashed = dict(root)
    del unhashed["bundle_sha256"]
    if root["bundle_sha256"] != canonical_sha256(unhashed):
        raise PolicyError("bundle_sha256 mismatch")

    policy = _exact(root["policy"], {
        "schema_version", "policy_kind", "target_manifest_refs", "revisions",
        "resource_classes", "model_classes", "output_namespace", "trials_per_strategy",
        "retry_policy", "evaluator", "calibration_policy", "method_spaces",
        "matrix", "baselines",
    }, "policy")
    if policy["schema_version"] != SCHEMA_VERSION or policy["policy_kind"] != POLICY_KIND:
        raise PolicyError("policy schema/kind mismatch")
    if not isinstance(policy["baselines"], Mapping) or not policy["baselines"]:
        raise PolicyError("policy baselines must be a non-empty object")
    objects = root["objects"]
    if not isinstance(objects, list) or not objects:
        raise PolicyError("objects must be a non-empty list")
    baseline_names: set[str] = set()
    for index, binding in enumerate(objects):
        item = _exact(binding, {"ref", "payload"}, f"objects[{index}]")
        payload = _exact(item["payload"], {
            "schema_version", "config_kind", "target_id", "arm_name", "method", "parameters",
        }, f"objects[{index}].payload")
        baseline_names.add(_string(payload["arm_name"], f"objects[{index}].payload.arm_name"))
    if len(baseline_names) != 1:
        raise PolicyError("every target must use one exact baseline name")
    baseline_name = next(iter(baseline_names))

    expected = build_policy_bundle(
        root["target_manifest_refs"], root["pair_text_refs"], policy["revisions"],
        policy["resource_classes"], policy["model_classes"], policy["output_namespace"],
        policy["trials_per_strategy"], policy["retry_policy"], policy["evaluator"],
        calibration_policy=policy["calibration_policy"],
        baseline_name=baseline_name,
    )
    actual_core = {key: value for key, value in policy.items() if key != "baselines"}
    expected_core = {key: value for key, value in expected["policy"].items() if key != "baselines"}
    if actual_core != expected_core:
        raise PolicyError("policy method space, matrix, revision, resource, or option drift")
    if root["target_manifest_refs"] != expected["target_manifest_refs"]:
        raise PolicyError("target_manifest_refs are not canonical")
    if root["pair_text_refs"] != expected["pair_text_refs"]:
        raise PolicyError("pair_text_refs are not canonical")
    if len(objects) != len(expected["objects"]):
        raise PolicyError("objects must contain exactly one baseline per target")

    expected_payloads = {item["payload"]["target_id"]: item["payload"] for item in expected["objects"]}
    actual_ref_by_target: dict[str, dict[str, str]] = {}
    seen_refs: set[tuple[str, str, str, str]] = set()
    for index, binding in enumerate(objects):
        payload = binding["payload"]
        target_id = payload["target_id"]
        if target_id in actual_ref_by_target or target_id not in expected_payloads or payload != expected_payloads[target_id]:
            raise PolicyError(f"objects[{index}] baseline payload drifted or duplicated")
        ref = validate_artifact_binding(binding["ref"], payload, f"objects[{index}].ref")
        key = (ref["uri"], ref["generation"], ref["size"], ref["sha256"])
        if key in seen_refs:
            raise PolicyError("objects contains a duplicate ref")
        seen_refs.add(key)
        actual_ref_by_target[target_id] = ref
        if allow_local_baselines:
            if not ref["uri"].startswith("bundle:///objects/") or ref["generation"] != "local":
                raise PolicyError("local policy object must use bundle:///objects/ and local generation")
        elif not ref["uri"].startswith("gs://") or not ref["generation"].isdigit() or int(ref["generation"]) < 1:
            raise PolicyError("production policy objects require generation-pinned gs:// refs")
    if set(actual_ref_by_target) != set(expected_payloads) or set(policy["baselines"]) != set(expected_payloads):
        raise PolicyError("baseline target coverage is incomplete")
    for target_id, ref in actual_ref_by_target.items():
        if validate_artifact_ref(policy["baselines"][target_id], f"policy.baselines.{target_id}") != ref:
            raise PolicyError(f"policy baseline ref for {target_id} differs from its object")
    if production or not allow_local_baselines:
        def require_production_ref(value: Any, label: str) -> None:
            source_ref = validate_artifact_ref(value, label)
            if not source_ref["uri"].startswith("gs://") or not source_ref["generation"].isdigit() or int(source_ref["generation"]) < 1:
                raise PolicyError(f"{label} must be a generation-pinned gs:// ref for production")
        validate_production_output_namespace(policy["output_namespace"], "policy.output_namespace")
        if policy["revisions"]["runtime"] != policy["revisions"]["code"]:
            raise PolicyError("production policy revisions.runtime must equal revisions.code")

        for index, binding in enumerate(root["target_manifest_refs"]):
            require_production_ref(binding["ref"], f"target_manifest_refs[{index}].ref")
            manifest = binding["payload"]
            require_production_ref(manifest["support"]["pair_texts_ref"], f"target_manifest_refs[{index}].support.pair_texts_ref")
            for route_index, route in enumerate(manifest["activation"]["routes"]):
                require_production_ref(route["completion_ref"], f"target_manifest_refs[{index}].routes[{route_index}].completion_ref")
                require_production_ref(route["proof_ref"], f"target_manifest_refs[{index}].routes[{route_index}].proof_ref")
        for target_id, pair_ref in root["pair_text_refs"].items():
            require_production_ref(pair_ref, f"pair_text_refs.{target_id}")
        for class_name, resource in policy["resource_classes"].items():
            require_production_ref(
                resource["dependency_lock_ref"],
                f"resource_classes.{class_name}.dependency_lock_ref",
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Canonical JSON generator inputs")
    parser.add_argument("--output", type=Path, required=True, help="Output policy bundle JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ContractError("spec must be a JSON object")
    bundle = build_policy_bundle(**spec)
    validate_policy_bundle(bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(bundle) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
