#!/usr/bin/env python3
"""Fail-closed compute runner for a frozen desired-results steering job."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

METHODS = frozenset({"caa", "ostrze", "mlp", "tecza", "tetno", "grom", "nurt", "wicher"})
STRATEGIES = (
    "chat_first", "chat_last", "chat_mean", "chat_max_norm", "chat_weighted",
    "mc_balanced", "role_play",
)
TRIALS_PER_FORMAT = 2
CALIBRATION_ROUTE_COUNT = len(METHODS) * len(STRATEGIES) * TRIALS_PER_FORMAT
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BENCHMARK = "winogrande"
BOUNDED_PROTOCOL_ID = "desired-results-bounded-rerun-v1"
BOUNDED_PROTOCOL_REVISION = 1
RUN_CLASS = "bounded_calibration_rerun"
BASE_SEED = 0
BOUNDED_OUTPUT_LEAF = "bounded-rerun-v1/hpo"
LAYERS = list(range(1, 17))
BASELINE_SPACE_HASHES = {
    "caa": "ee24b89708ec69c05954a8117e9e0478605c12d3c09ebad5d945f36dd6c5b7c8",
    "ostrze": "e945f8ec6214d88f482224d80200522f2060fcedd49835da88ffc1c12d99ecc1",
    "mlp": "c96fb04e95b1e7512b45ded93708c829a23a7ddf66a06953cda782ff9f63b60a",
    "tecza": "0a301d86a4bb1e1b1c3cf459d1569a3409e9dca8ea19548384387b4f52f13f25",
    "tetno": "f64aeb672534dcbbfecc742f3a42ae836fbef1959897d70c7584fb076a57d4eb",
    "grom": "cfd9f7c75f6459caf4b06313c854c98001b682aaf7c36da97193eaae4ecb7101",
    "nurt": "515d5c7c8f78e902a8d408608dd24067956e2afee114c75e80693a134881bfa7",
    "wicher": "bbc2cc14ad789800ec8dbc48b662a43777f1037dd3277d367eafbde466185b51",
}


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_param(param: Any) -> Dict[str, Any]:
    from wisent.core.utils.services.optimization.core.parameters import (
        CategoricalParam, FloatParam, IntParam,
    )
    if isinstance(param, CategoricalParam):
        return {"kind": "categorical", "choices": list(param.choices)}
    if isinstance(param, FloatParam):
        if param.distribution != "uniform":
            raise ContractError(f"unbounded float distribution {param.distribution!r}")
        return {"kind": "float", "distribution": "uniform", "low": param.low,
                "high": param.high, "log_scale": param.log_scale}
    if isinstance(param, IntParam):
        if param.distribution != "randint":
            raise ContractError(f"unbounded integer distribution {param.distribution!r}")
        return {"kind": "int", "distribution": "randint", "low": param.low,
                "high": param.high}
    raise ContractError(f"unsupported parameter object {type(param).__name__}")


def _bounded_overrides() -> Dict[str, Any]:
    # This is the sole owner of calibration-local changes to Wisent's global spaces.
    return json.loads('''{"caa":{},"common":{"strength":{"distribution":"uniform","high":4,"kind":"float","log_scale":true,"low":0.25}},"grom":{"adapt_complex_directions":{"choices":[5],"kind":"categorical"},"adapt_linear_directions":{"choices":[1],"kind":"categorical"},"adapt_max_directions":{"choices":[8],"kind":"categorical"},"behavior_weight":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.25},"caa_alignment_weight":{"distribution":"uniform","high":1,"kind":"float","log_scale":true,"low":0.001},"concentration_weight":{"distribution":"uniform","high":1,"kind":"float","log_scale":true,"low":0.001},"contrastive_weight":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.1},"create_noise_scale":{"distribution":"uniform","high":0.1,"kind":"float","log_scale":true,"low":0.0001},"gate_dim_divisor":{"choices":[4],"kind":"categorical"},"gate_dim_max":{"choices":[128],"kind":"categorical"},"gate_dim_min":{"choices":[128],"kind":"categorical"},"gate_hidden_dim":{"choices":[128],"kind":"categorical"},"gate_temperature":{"distribution":"uniform","high":1,"kind":"float","log_scale":true,"low":0.05},"gate_warmup_weight":{"distribution":"uniform","high":1,"kind":"float","log_scale":true,"low":0.001},"independence_weight":{"distribution":"uniform","high":0.2,"kind":"float","log_scale":true,"low":0.001},"intensity_dim_divisor":{"choices":[4],"kind":"categorical"},"intensity_dim_max":{"choices":[64],"kind":"categorical"},"intensity_dim_min":{"choices":[64],"kind":"categorical"},"intensity_hidden_dim":{"choices":[64],"kind":"categorical"},"learning_rate":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":0.0001},"max_alpha":{"distribution":"uniform","high":4,"kind":"float","log_scale":true,"low":0.5},"max_cosine_sim":{"choices":[0.9],"kind":"categorical"},"max_grad_norm":{"distribution":"uniform","high":5,"kind":"float","log_scale":true,"low":0.5},"min_adapted_directions":{"choices":[1],"kind":"categorical"},"min_cosine_sim":{"choices":[-0.1],"kind":"categorical"},"num_directions":{"distribution":"randint","high":8,"kind":"int","low":2},"optimization_steps":{"distribution":"randint","high":300,"kind":"int","low":100},"sensor_layer":{"choices":[8],"kind":"categorical"},"significant_directions_default":{"choices":[3],"kind":"categorical"},"smooth_weight":{"distribution":"uniform","high":0.2,"kind":"float","log_scale":true,"low":0.001},"sparse_weight":{"distribution":"uniform","high":0.2,"kind":"float","log_scale":true,"low":0.001},"steering_end":{"choices":[8],"kind":"categorical"},"steering_start":{"choices":[8],"kind":"categorical"},"utility_weight":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.1},"warmup_steps":{"distribution":"randint","high":50,"kind":"int","low":10},"weight_decay":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":1e-06}},"mlp":{"early_stop_tol":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":1e-05},"epochs":{"distribution":"randint","high":150,"kind":"int","low":20},"hidden_dim":{"distribution":"randint","high":256,"kind":"int","low":32},"learning_rate":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":0.0001},"weight_decay":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":1e-06}},"nurt":{"lr":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":0.0001},"lr_min":{"distribution":"uniform","high":0.0001,"kind":"float","log_scale":true,"low":1e-06},"max_concept_dim":{"choices":[32],"kind":"categorical"},"max_grad_norm":{"distribution":"uniform","high":5,"kind":"float","log_scale":true,"low":0.5},"num_dims":{"distribution":"randint","high":32,"kind":"int","low":0},"training_epochs":{"distribution":"randint","high":300,"kind":"int","low":50},"weight_decay":{"distribution":"uniform","high":0.01,"kind":"float","log_scale":true,"low":1e-06}},"ostrze":{"C":{"distribution":"uniform","high":100,"kind":"float","log_scale":true,"low":0.001}},"tecza":{"ablation_weight":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.001},"addition_weight":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.001},"independence_weight":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.001},"learning_rate":{"distribution":"uniform","high":0.05,"kind":"float","log_scale":true,"low":0.0001},"max_cosine_similarity":{"choices":[0.9],"kind":"categorical"},"max_directions":{"choices":[10],"kind":"categorical"},"min_cosine_similarity":{"choices":[-0.1],"kind":"categorical"},"num_directions":{"distribution":"randint","high":10,"kind":"int","low":1},"optimization_steps":{"distribution":"randint","high":300,"kind":"int","low":50},"perturbation_scale":{"distribution":"uniform","high":0.3,"kind":"float","log_scale":true,"low":0.001},"universal_basis_noise":{"distribution":"uniform","high":0.1,"kind":"float","log_scale":true,"low":0.0001}},"tetno":{"entropy_ceiling":{"choices":[5],"kind":"categorical"},"entropy_floor":{"choices":[1],"kind":"categorical"},"gate_temperature":{"distribution":"uniform","high":2,"kind":"float","log_scale":true,"low":0.05},"learning_rate":{"distribution":"uniform","high":0.05,"kind":"float","log_scale":true,"low":0.0001},"max_alpha":{"distribution":"uniform","high":4,"kind":"float","log_scale":true,"low":0.25},"optimization_steps":{"distribution":"randint","high":300,"kind":"int","low":50},"sensor_layer":{"choices":[8],"kind":"categorical"},"steering_end":{"choices":[8],"kind":"categorical"},"steering_start":{"choices":[8],"kind":"categorical"}},"wicher":{"alpha":{"distribution":"uniform","high":1,"kind":"float","log_scale":true,"low":0.0001}}}''')


def _raw_param(param: Any) -> Dict[str, Any]:
    return asdict(param)


def _bounded_prior_definitions() -> Dict[str, Dict[str, Any]]:
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.search_space import get_method_space
    definitions = {}
    overrides = _bounded_overrides()
    for method in sorted(METHODS):
        baseline = get_method_space(method, 17)
        raw = {name: _raw_param(param) for name, param in sorted(baseline.items())}
        if _canonical_json_sha256(raw) != BASELINE_SPACE_HASHES[method]:
            raise ContractError(f"global {method} search space drifted from the frozen baseline")
        specs = {}
        for name, param in baseline.items():
            try:
                specs[name] = _serialize_param(param)
            except ContractError:
                pass
        specs.update(overrides["common"])
        specs.update(overrides[method])
        specs["extraction_strategy"] = {"kind": "categorical", "choices": list(STRATEGIES)}
        if "layer" in baseline:
            specs["layer"] = {"kind": "categorical", "choices": list(LAYERS)}
        if set(specs) != set(baseline):
            raise ContractError(f"bounded {method} definitions do not cover the complete global space")
        definitions[method] = {name: specs[name] for name in sorted(specs)}
    _validate_prior_definitions(definitions, LAYERS)
    return definitions


def _validate_param_definition(name: str, spec: Any) -> None:
    if not isinstance(spec, dict) or spec.get("kind") not in {"float", "int", "categorical"}:
        raise ContractError(f"{name} has an unsupported parameter definition")
    kind = spec["kind"]
    if kind == "categorical":
        if set(spec) != {"kind", "choices"} or not isinstance(spec["choices"], list) or not spec["choices"]:
            raise ContractError(f"{name} has an invalid categorical definition")
        encoded = [json.dumps(v, sort_keys=True, allow_nan=False) for v in spec["choices"]]
        if len(encoded) != len(set(encoded)):
            raise ContractError(f"{name} has duplicate categorical choices")
        return
    expected = {"kind", "distribution", "low", "high"}
    if kind == "float":
        expected.add("log_scale")
    if set(spec) != expected or spec.get("distribution") != ("uniform" if kind == "float" else "randint"):
        raise ContractError(f"{name} has an invalid bounded numeric definition")
    low, high = spec["low"], spec["high"]
    if isinstance(low, bool) or isinstance(high, bool) or not all(isinstance(v, (int, float)) for v in (low, high)):
        raise ContractError(f"{name} bounds must be numbers, not booleans")
    if not all(math.isfinite(v) for v in (low, high)) or low > high:
        raise ContractError(f"{name} bounds must be finite and ordered")
    if kind == "int" and (type(low) is not int or type(high) is not int):
        raise ContractError(f"{name} randint bounds must be integral")
    if kind == "float" and (type(spec["log_scale"]) is not bool or spec["log_scale"] and low <= 0):
        raise ContractError(f"{name} has invalid log-scale bounds")


def _support(spec: Mapping[str, Any]) -> Tuple[Any, Any]:
    if spec["kind"] == "categorical":
        return min(spec["choices"]), max(spec["choices"])
    return spec["low"], spec["high"]


def _validate_dependency_contract(method: str, specs: Mapping[str, Any], layers: Sequence[int]) -> None:
    lo = lambda name: _support(specs[name])[0]
    hi = lambda name: _support(specs[name])[1]
    if method == "tecza" and not (hi("num_directions") <= lo("max_directions") and hi("min_cosine_similarity") <= lo("max_cosine_similarity")):
        raise ContractError("TECZA dependent supports overlap invalidly")
    if method == "tetno" and not (hi("entropy_floor") < lo("entropy_ceiling") and specs["sensor_layer"] == specs["steering_start"] == specs["steering_end"]):
        raise ContractError("TETNO dependent supports overlap invalidly")
    if method == "grom" and not (hi("warmup_steps") < lo("optimization_steps") and specs["sensor_layer"] == specs["steering_start"] == specs["steering_end"] and hi("min_cosine_sim") <= lo("max_cosine_sim") and hi("gate_dim_min") <= lo("gate_dim_max") and hi("intensity_dim_min") <= lo("intensity_dim_max") and hi("adapt_linear_directions") <= lo("adapt_complex_directions") <= hi("adapt_complex_directions") <= lo("adapt_max_directions")):
        raise ContractError("GROM dependent supports overlap invalidly")
    if method == "nurt" and not (hi("num_dims") <= lo("max_concept_dim") and hi("lr_min") <= lo("lr")):
        raise ContractError("NURT dependent supports overlap invalidly")


def _validate_prior_definitions(definitions: Any, layers: Sequence[int]) -> None:
    if not isinstance(definitions, dict) or set(definitions) != METHODS or list(layers) != LAYERS:
        raise ContractError("prior definitions require all eight methods and exact layers 1..16")
    for method, specs in definitions.items():
        if not isinstance(specs, dict) or not specs:
            raise ContractError(f"{method} prior must be a non-empty object")
        for name, spec in specs.items():
            _validate_param_definition(name, spec)
        _validate_dependency_contract(method, specs, layers)


def _study_seed(base_seed: int, method: str, strategy: str) -> int:
    material = f"{BOUNDED_PROTOCOL_ID}\0{base_seed}\0{method}\0{strategy}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big", signed=False)


def _calibration_plan(trials_per_format: int = TRIALS_PER_FORMAT) -> Dict[str, Any]:
    if trials_per_format != TRIALS_PER_FORMAT:
        raise ContractError(f"calibration requires exactly {TRIALS_PER_FORMAT} trials per format")
    priors = _bounded_prior_definitions()
    routes = []
    for method in sorted(METHODS):
        for strategy in STRATEGIES:
            seed = _study_seed(BASE_SEED, method, strategy)
            for repeat in range(trials_per_format):
                route_key = f"{method}:{strategy}:repeat-{repeat}"
                routes.append({"method": method, "extraction_strategy": strategy,
                               "repeat": repeat, "run_key": route_key,
                               "staging_prefix": f"calibration/{method}/{strategy}/repeat-{repeat}/",
                               "protocol_id": BOUNDED_PROTOCOL_ID, "study_seed": seed,
                               "test_enabled": False})
    return {
        "schema_version": 2,
        "protocol_identity": {"id": BOUNDED_PROTOCOL_ID, "revision": BOUNDED_PROTOCOL_REVISION,
            "run_class": RUN_CLASS, "model": MODEL, "benchmark": BENCHMARK,
            "extraction_component": "residual_stream"},
        "optimizer_contract": {"backend": "optuna", "sampler": "random", "pruner": "nop",
            "base_seed": BASE_SEED, "seed_algorithm": "sha256-first-u32-be-v1",
            "integer_bounds": "inclusive", "load_if_exists": False, "extra_trials": 0,
            "trials_per_format": TRIALS_PER_FORMAT},
        "data_contract": {"fit_splits": ["train"], "selection_split": "validation",
            "final_fit_splits": ["train"], "test_evaluations": 0, "layers": LAYERS},
        "prior_definitions": priors,
        "prior_definitions_sha256": _canonical_json_sha256(priors),
        "invalid_exploration_disposition": {"protocol_id": "steering_effectiveness_initial",
            "status": "invalid_unbounded_exploration", "eligible_for_selection": False,
            "consumed_as_prior_or_resume": False},
        "trials_per_format": trials_per_format, "route_count": len(routes), "routes": routes,
    }


def _validate_calibration_plan(plan: Any, trials_per_format: int) -> Dict[str, Any]:
    expected = _calibration_plan(trials_per_format)
    if plan != expected:
        raise ContractError("calibration plan must equal the exact canonical schema-v2 plan")
    if plan["prior_definitions_sha256"] != _canonical_json_sha256(plan["prior_definitions"]):
        raise ContractError("calibration prior hash is invalid")
    _validate_prior_definitions(plan["prior_definitions"], plan["data_contract"]["layers"])
    routes = plan["routes"]
    expected_cross_product = {(m, s, r) for m in METHODS for s in STRATEGIES for r in range(TRIALS_PER_FORMAT)}
    actual = {(r["method"], r["extraction_strategy"], r["repeat"]) for r in routes}
    if len(routes) != CALIBRATION_ROUTE_COUNT or actual != expected_cross_product:
        raise ContractError("calibration plan does not contain the exact 112-route cross product")
    if len({r["run_key"] for r in routes}) != len(routes) or len({r["staging_prefix"] for r in routes}) != len(routes):
        raise ContractError("calibration plan route identities must be unique")
    for route in routes:
        if route["study_seed"] != _study_seed(BASE_SEED, route["method"], route["extraction_strategy"]):
            raise ContractError("calibration route contains an inconsistent study seed")
        if route.get("test_enabled") is not False or "test_pair_ids" in route:
            raise ContractError("calibration routes must exclude test evaluation")
    return plan




class ContractError(RuntimeError):
    """The frozen execution contract is incomplete or inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON from {path}: {exc}") from exc


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _calibration_protocol() -> Dict[str, Any]:
    return {
        "id": BOUNDED_PROTOCOL_ID, "revision": BOUNDED_PROTOCOL_REVISION,
        "run_class": RUN_CLASS,
        "prior_owner": "scripts/steering/desired_results_runner.py",
        "methods": sorted(METHODS), "extraction_component": "residual_stream",
        "extraction_strategies": list(STRATEGIES), "trials_per_format": 2,
        "format_count": 7, "trials_per_method": 14,
        "selection_split": "validation", "fit_splits": ["train"],
        "final_fit_splits": ["train"], "test_evaluations": 0,
        "exploratory_run_disposition": "invalid_unbounded_priors_excluded",
    }


def _manifest(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ContractError("preflight manifest must use schema_version 1")
    if data.get("purpose") != "calibration" or data.get("execution_mode") != "calibration":
        raise ContractError("runner requires a calibration-purpose manifest")
    if data.get("calibration_protocol") != _calibration_protocol():
        raise ContractError("manifest calibration_protocol is not the exact bounded contract")
    unit = data.get("job_unit", {})
    if unit.get("model") != MODEL or unit.get("benchmark") != BENCHMARK:
        raise ContractError("preflight manifest is not the pinned Llama-3.2-1B-Instruct x winogrande target")
    method = unit.get("method")
    if method not in METHODS:
        raise ContractError(f"method {method!r} is not in the eight-method calibration scope")
    scope = data.get("activation_search_scope", {})
    if scope.get("extraction_component") != "residual_stream":
        raise ContractError("only residual_stream activation input is permitted")
    if tuple(scope.get("extraction_strategies", ())) != STRATEGIES:
        raise ContractError("preflight manifest does not contain the frozen seven-format scope")
    layers = scope.get("layers")
    if layers != LAYERS:
        raise ContractError("calibration manifest layers must be exactly 1 through 16")
    split = data.get("split", {})
    if set(split) != {"counts", "pair_ids", "hpo_reads", "selection_split", "final_fit", "test_evaluations"}:
        raise ContractError("calibration split contract contains missing or extra fields")
    pair_sets = split.get("pair_ids", {})
    if set(pair_sets) != {"train", "validation"}:
        raise ContractError(
            "calibration manifest must contain exactly train and validation pair IDs; test IDs are forbidden"
        )
    normalized = {}
    for name in ("train", "validation"):
        values = pair_sets[name]
        if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
            raise ContractError(f"split {name} must contain integer pair IDs")
        if len(values) != len(set(values)):
            raise ContractError(f"split {name} contains duplicate pair IDs")
        normalized[name] = set(values)
    if split.get("counts") != {name: len(pair_sets[name]) for name in ("train", "validation")}:
        raise ContractError("calibration split counts do not match pair IDs")
    if normalized["train"] & normalized["validation"]:
        raise ContractError("train and validation supports must be disjoint")
    if (split.get("hpo_reads") != ["train"] or
            split.get("selection_split") != "validation" or
            split.get("final_fit") != ["train"] or split.get("test_evaluations") != 0):
        raise ContractError("calibration must read/fit train, select validation, and never evaluate test")
    output_prefix = data.get("output_prefix")
    if not isinstance(output_prefix, str) or not output_prefix:
        raise ContractError("calibration manifest requires an output_prefix")
    expected_writes = f"{output_prefix}bounded-rerun-v1/hpo/"
    hpo_contract = data.get("mode_contracts", {}).get("hpo", {})
    if set(data.get("mode_contracts", {})) != {"hpo"} or hpo_contract.get("writes_under") != expected_writes:
        raise ContractError("calibration HPO must write only under bounded-rerun-v1/hpo")
    forbidden_test_keys = {"test_pair_ids", "final_test_reads", "final_test"}
    if forbidden_test_keys & set(data):
        raise ContractError("calibration manifest contains forbidden test fields")
    policy = data.get("saved_activation_policy", {})
    if policy.get("automatic_regeneration") != "forbidden" or policy.get("fallback") != "forbidden" or policy.get("positional_join") != "forbidden":
        raise ContractError("strict saved-activation policy is missing")
    return data


def _completion_routes(index_path: Path, manifest: Mapping[str, Any]) -> Dict[Tuple[str, int], Path]:
    index = _read_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("artifacts"), list):
        raise ContractError("completion index must be an object with an artifacts list")
    expected = {(strategy, layer) for strategy in STRATEGIES for layer in manifest["activation_search_scope"]["layers"]}
    routes: Dict[Tuple[str, int], Path] = {}
    full_support = set().union(*manifest["split"]["pair_ids"].values())
    for record in index["artifacts"]:
        if not isinstance(record, dict):
            raise ContractError("completion index artifact entries must be objects")
        key = (record.get("extraction_strategy"), record.get("layer"))
        if key not in expected:
            raise ContractError(f"completion index contains out-of-scope route {key!r}")
        if key in routes:
            raise ContractError(f"completion index contains duplicate route {key!r}")
        raw_path = record.get("completion_manifest")
        if not isinstance(raw_path, str) or not raw_path:
            raise ContractError(f"completion index route {key!r} has no completion_manifest path")
        proof_path = Path(raw_path)
        if not proof_path.is_absolute():
            proof_path = index_path.parent / proof_path
        proof_path = proof_path.resolve()
        proof = _read_json(proof_path)
        status = proof.get("complete", proof.get("status")) if isinstance(proof, dict) else None
        if status not in (True, "complete", "completed"):
            raise ContractError(f"completion proof for route {key!r} is not complete")
        proof_strategy = proof.get("extraction_strategy", proof.get("strategy", proof.get("format")))
        proof_layers = proof.get("layers", [proof.get("layer")] if "layer" in proof else None)
        if proof_strategy != key[0] or proof_layers != [key[1]]:
            raise ContractError(f"completion proof identity does not match route {key!r}")
        if proof.get("model", proof.get("model_name", MODEL)) != MODEL:
            raise ContractError(f"completion proof model does not match route {key!r}")
        if proof.get("benchmark", proof.get("task_name", proof.get("task", BENCHMARK))) != BENCHMARK:
            raise ContractError(f"completion proof benchmark does not match route {key!r}")
        proof_ids = proof.get("pair_ids", proof.get("expected_pair_ids", proof.get("support_pair_ids")))
        if proof_ids is None and isinstance(proof.get("support"), dict):
            proof_ids = proof["support"].get("pair_ids", proof["support"].get("expected_pair_ids"))
        if (
            not isinstance(proof_ids, list)
            or any(type(value) is not int for value in proof_ids)
            or len(proof_ids) != len(set(proof_ids))
        ):
            raise ContractError(f"completion proof for route {key!r} has invalid pair support")
        missing_support = sorted(full_support.difference(proof_ids))
        if missing_support:
            raise ContractError(
                f"completion proof for route {key!r} is missing calibration support: {missing_support}"
            )
        routes[key] = proof_path
    missing = sorted(expected.difference(routes))
    if missing:
        raise ContractError(f"completion index is missing routes: {missing}")
    return routes


def _pair_file(path: Path, pair_ids: Sequence[int]) -> None:
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import load_pair_texts_from_hf_strict
    texts = load_pair_texts_from_hf_strict(BENCHMARK, pair_ids)
    pairs = [{
        "pair_id": pair_id,
        "prompt": texts[pair_id]["prompt"],
        "positive_response": {"model_response": texts[pair_id]["positive"]},
        "negative_response": {"model_response": texts[pair_id]["negative"]},
    } for pair_id in pair_ids]
    _atomic_json(path, {"task_name": BENCHMARK, "num_pairs": len(pairs), "pair_ids": list(pair_ids), "pairs": pairs})


def _materialize(root: Path, pair_ids: Sequence[int], routes: Mapping[Tuple[str, int], Path]) -> Dict[Tuple[str, int], str]:
    from wisent.core.reading.modules.utilities.data.enriched_builder import build_enriched_from_hf_strict
    outputs = {}
    for (strategy, layer), proof in routes.items():
        route_dir = root / strategy / f"layer_{layer}"
        route_dir.mkdir(parents=True)
        outputs[(strategy, layer)] = build_enriched_from_hf_strict(
            MODEL, BENCHMARK, layer, strategy, str(route_dir), pair_ids,
            completion_manifest_file=str(proof),
        )
    return outputs


def _destination(args: argparse.Namespace, manifest: Mapping[str, Any], leaf: str) -> Path:
    prefix = manifest.get("output_prefix")
    if not isinstance(prefix, str) or Path(prefix).is_absolute() or ".." in Path(prefix).parts:
        raise ContractError("manifest output_prefix is not a safe relative staging prefix")
    return (args.output_root.resolve() / prefix / leaf).resolve()


def _new_staging(destination: Path) -> Path:
    if destination.exists():
        raise ContractError(f"destination already exists; refusing repeat execution: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))


def _publish(staging: Path, destination: Path) -> None:
    try:
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _param_from_definition(spec: Mapping[str, Any]):
    from wisent.core.utils.services.optimization.core.parameters import (
        CategoricalParam, FloatParam, IntParam,
    )
    if spec["kind"] == "categorical":
        return CategoricalParam(choices=list(spec["choices"]))
    if spec["kind"] == "float":
        return FloatParam(distribution="uniform", low=spec["low"], high=spec["high"],
                          log_scale=spec["log_scale"])
    if spec["kind"] == "int":
        # Active Optuna suggest_int uses both endpoints inclusively; no adjustment is made.
        return IntParam(distribution="randint", low=spec["low"], high=spec["high"])
    raise ContractError(f"unsupported parameter kind {spec.get('kind')!r}")


def _search_space(method: str, layers: Sequence[int], strategy: str,
                  plan: Mapping[str, Any]):
    if method not in METHODS or strategy not in STRATEGIES or list(layers) != LAYERS:
        raise ContractError("search-space route is outside the frozen method/format/layer scope")
    canonical = plan["prior_definitions"][method]
    current = _bounded_prior_definitions()[method]
    if current != canonical:
        raise ContractError(f"bounded {method} search space differs from the frozen plan")
    route_specs = {name: dict(spec) for name, spec in canonical.items()}
    route_specs["extraction_strategy"] = {"kind": "categorical", "choices": [strategy]}
    space = {name: _param_from_definition(route_specs[name]) for name in sorted(route_specs)}
    if {name: _serialize_param(param) for name, param in space.items()} != route_specs:
        raise ContractError("constructed local search space failed canonical serialization")
    return space


def _validate_sample(method: str, params: Any, specs: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict) or set(params) != set(specs):
        raise ContractError(f"{method} sample keys differ from the frozen prior")
    for name, spec in specs.items():
        value = params[name]
        if spec["kind"] == "categorical":
            if value not in spec["choices"]:
                raise ContractError(f"{method}.{name} is outside categorical support")
        elif spec["kind"] == "int":
            if type(value) is not int or not spec["low"] <= value <= spec["high"]:
                raise ContractError(f"{method}.{name} is outside inclusive randint support")
        elif (isinstance(value, bool) or not isinstance(value, (int, float)) or
              not math.isfinite(value) or not spec["low"] <= value <= spec["high"]):
            raise ContractError(f"{method}.{name} is outside finite float support")
    if method == "tecza" and not (params["num_directions"] <= params["max_directions"] and params["min_cosine_similarity"] <= params["max_cosine_similarity"]):
        raise ContractError("invalid TECZA dependent sample")
    if method == "tetno" and not (params["entropy_floor"] < params["entropy_ceiling"] and params["sensor_layer"] == params["steering_start"] == params["steering_end"]):
        raise ContractError("invalid TETNO dependent sample")
    if method == "grom" and not (params["warmup_steps"] < params["optimization_steps"] and params["sensor_layer"] == params["steering_start"] == params["steering_end"] and params["min_cosine_sim"] <= params["max_cosine_sim"] and params["gate_dim_min"] <= params["gate_dim_max"] and params["intensity_dim_min"] <= params["intensity_dim_max"] and params["adapt_linear_directions"] <= params["adapt_complex_directions"] <= params["adapt_max_directions"]):
        raise ContractError("invalid GROM dependent sample")
    if method == "nurt" and not (params["num_dims"] <= params["max_concept_dim"] and params["lr_min"] <= params["lr"]):
        raise ContractError("invalid NURT dependent sample")
    return params


def _finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{label} must be a finite numeric score")
    return float(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_hpo(args: argparse.Namespace, manifest_path: Path, manifest: Mapping[str, Any],
             routes, plan: Mapping[str, Any]) -> Path:
    method = manifest["job_unit"]["method"]
    destination = _destination(args, manifest, BOUNDED_OUTPUT_LEAF)
    staging = _new_staging(destination)
    try:
        train_ids = manifest["split"]["pair_ids"]["train"]
        validation_ids = manifest["split"]["pair_ids"]["validation"]
        enriched = _materialize(staging / "strict_train", train_ids, routes)
        validation_file = staging / "validation_pairs.json"
        _pair_file(validation_file, validation_ids)
        from wisent.core.utils.cli.commands.optimize_steering.pipeline.pipeline import create_objective
        from wisent.core.utils.services.optimization.core.atoms import BaseOptimizer, HPOConfig
        from wisent.core.primitives.models.wisent_model import WisentModel
        cached_model = WisentModel(MODEL, device=args.device)
        optimizer = BaseOptimizer()
        optimizer.direction = "maximize"
        format_results = []
        layers = manifest["activation_search_scope"]["layers"]
        study_seeds = {}
        for strategy in STRATEGIES:
            work_dir = staging / "trial_work" / strategy
            work_dir.mkdir(parents=True)
            real_objective = create_objective(
                method=method, model=MODEL, task=BENCHMARK,
                num_layers=max(layers) + 1, limit=None, device=args.device,
                work_dir=str(work_dir), test_pairs_file=str(validation_file),
                strict_enriched_files=enriched, cached_model=cached_model,
            )
            route_specs = {name: dict(spec) for name, spec in plan["prior_definitions"][method].items()}
            route_specs["extraction_strategy"] = {"kind": "categorical", "choices": [strategy]}
            def objective(params, _real=real_objective, _specs=route_specs):
                _validate_sample(method, params, _specs)
                return _finite_score(_real(params), f"{strategy} objective")
            seed = next(route["study_seed"] for route in plan["routes"]
                        if route["method"] == method and route["extraction_strategy"] == strategy)
            study_seeds[strategy] = seed
            result = optimizer.optimize_fn(
                objective, _search_space(method, layers, strategy, plan), TRIALS_PER_FORMAT,
                cfg=HPOConfig(backend="optuna", n_trials=TRIALS_PER_FORMAT, seed=seed,
                              sampler="random", pruner="nop", load_if_exists=False),
                extra_trials=0,
            )
            if result.backend != "optuna" or result.n_trials != TRIALS_PER_FORMAT or len(result.all_trials) != TRIALS_PER_FORMAT:
                raise ContractError(f"format {strategy!r} did not return exactly two successful Optuna trials")
            observed = []
            for trial in result.all_trials:
                if not isinstance(trial, dict) or set(trial) != {"params", "score"}:
                    raise ContractError(f"format {strategy!r} returned a malformed trial")
                params = dict(trial["params"])
                _validate_sample(method, params, route_specs)
                score = _finite_score(trial["score"], f"{strategy} trial score")
                observed.append({"params": params, "score": score})
            best_params = dict(result.best_params)
            _validate_sample(method, best_params, route_specs)
            best_score = _finite_score(result.best_score, f"{strategy} best score")
            if not any(item["params"] == best_params and item["score"] == best_score for item in observed):
                raise ContractError(f"format {strategy!r} best result is not an observed trial")
            format_results.append({"extraction_strategy": strategy, "trial_count": 2,
                                   "scores": [item["score"] for item in observed],
                                   "best_params": best_params,
                                   "best_validation_score": best_score, "trials": observed})
        if len(format_results) != len(STRATEGIES):
            raise ContractError("HPO did not preserve the seven-format budget")
        global_best = max(format_results, key=lambda item: item["best_validation_score"])
        global_params = dict(global_best["best_params"])
        provenance_core = {
            "protocol_identity": plan["protocol_identity"], "run_class": RUN_CLASS,
            "eligible_for_selection": True,
            "prior_definitions_sha256": plan["prior_definitions_sha256"],
            "old_exploratory_run": {"protocol_id": "steering_effectiveness_initial",
                                    "excluded": True, "consumed_as_prior_or_resume": False},
        }
        identity = {
            "schema_version": 2, "mode": "hpo", "model": MODEL, "benchmark": BENCHMARK,
            "method": method, **provenance_core,
            "manifest_sha256": _digest(manifest_path),
            "completion_index_sha256": _digest(args.completion_index.resolve()),
            "calibration_plan_sha256": _digest(args.calibration_plan.resolve()),
            "runner_sha256": _digest(Path(__file__).resolve()),
            "fit_pair_ids": train_ids, "selection_pair_ids": validation_ids,
            "test_pair_ids_read": [], "trials_per_format": TRIALS_PER_FORMAT,
            "trial_count": len(STRATEGIES) * TRIALS_PER_FORMAT,
            "per_format": format_results, "best_params": global_params,
            "best_validation_score": global_best["best_validation_score"],
        }
        provenance = {
            "schema_version": 1, **provenance_core,
            "hashes": {"prior_definitions": plan["prior_definitions_sha256"],
                       "plan": identity["calibration_plan_sha256"],
                       "manifest": identity["manifest_sha256"],
                       "completion_index": identity["completion_index_sha256"],
                       "runner": identity["runner_sha256"],
                       "fit_support": _canonical_json_sha256(train_ids),
                       "selection_support": _canonical_json_sha256(validation_ids)},
            "support": {"fit_split": "train", "fit_pair_ids": train_ids,
                        "selection_split": "validation", "selection_pair_ids": validation_ids,
                        "test_pair_ids_read": []},
            "optimizer": {**plan["optimizer_contract"], "study_seeds": study_seeds},
            "runtime": {"python": platform.python_version(), "optuna": _package_version("optuna"),
                        "torch": _package_version("torch"), "wisent": _package_version("wisent"),
                        "model_revision": manifest["revisions"]["model"],
                        "activation_revision": manifest["revisions"]["activation"]},
        }
        _atomic_json(staging / "best_config.json", global_params)
        _atomic_json(staging / "validation_summary.json", {
            "selection_split": "validation", "pair_ids": validation_ids,
            "trials_per_format": TRIALS_PER_FORMAT, "trial_count": 14,
            "per_format": [{key: item[key] for key in ("extraction_strategy", "trial_count", "scores", "best_params", "best_validation_score")} for item in format_results],
            "best_score": global_best["best_validation_score"], "best_params": global_params,
        })
        _atomic_json(staging / "trials.json", {"per_format": format_results})
        _atomic_json(staging / "frozen_config.json", identity)
        _atomic_json(staging / "provenance.json", provenance)
        os.chmod(staging / "frozen_config.json", 0o444)
        shutil.rmtree(staging / "strict_train")
        shutil.rmtree(staging / "trial_work")
        _publish(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise




def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contract", "hpo"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--completion-index", type=Path, required=True)
    parser.add_argument("--calibration-plan", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--trials-per-format", type=int, default=TRIALS_PER_FORMAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.seed != BASE_SEED:
            raise ContractError(f"--seed must equal protocol base seed {BASE_SEED}")
        plan = _calibration_plan(args.trials_per_format)
        manifest_path = args.manifest.resolve()
        manifest = _manifest(manifest_path)
        method = manifest["job_unit"]["method"]
        args.completion_index = args.completion_index.resolve()
        routes = _completion_routes(args.completion_index, manifest)
        if args.mode == "contract":
            if args.calibration_plan is not None:
                plan_path = args.calibration_plan.resolve()
                if plan_path.exists():
                    _validate_calibration_plan(_read_json(plan_path), args.trials_per_format)
                else:
                    _atomic_json(plan_path, plan)
            print(json.dumps({
                "status": "ok", "mode": "contract", "model_loaded": False,
                "method": method,
                "route_count": len(routes),
                "routes": [
                    {"extraction_strategy": strategy, "layer": layer}
                    for strategy, layer in sorted(routes)
                ],
                "fit_splits": ["train"], "selection_split": "validation",
                "final_fit_splits": ["train"], "test_evaluations": 0,
                "stratified_budget": {
                    "method_count": len(METHODS),
                    "format_count": len(STRATEGIES),
                    "trials_per_format": args.trials_per_format,
                    "trials_per_method": len(STRATEGIES) * args.trials_per_format,
                    "route_count": plan["route_count"],
                },
                "calibration_plan": plan,
            }, sort_keys=True))
            return 0
        if args.output_root is None:
            raise ContractError("--output-root is required")
        if args.calibration_plan is None or not args.calibration_plan.is_file():
            raise ContractError("HPO requires a materialized --calibration-plan from contract mode")
        args.calibration_plan = args.calibration_plan.resolve()
        plan = _validate_calibration_plan(_read_json(args.calibration_plan), args.trials_per_format)
        destination = _run_hpo(args, manifest_path, manifest, routes, plan)
        print(destination)
        return 0
    except (ContractError, ValueError, OSError) as exc:
        print(f"desired-results runner refused execution: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
