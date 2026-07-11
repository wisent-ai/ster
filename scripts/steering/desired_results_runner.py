#!/usr/bin/env python3
"""Fail-closed compute runner for a frozen desired-results steering job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
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


def _calibration_plan(trials_per_format: int = TRIALS_PER_FORMAT) -> Dict[str, Any]:
    if trials_per_format != TRIALS_PER_FORMAT:
        raise ContractError(
            f"calibration requires exactly {TRIALS_PER_FORMAT} trials per format"
        )
    routes = []
    for method in sorted(METHODS):
        for strategy in STRATEGIES:
            for repeat in range(trials_per_format):
                route_key = f"{method}:{strategy}:repeat-{repeat}"
                routes.append({
                    "method": method,
                    "extraction_strategy": strategy,
                    "repeat": repeat,
                    "run_key": route_key,
                    "staging_prefix": f"calibration/{method}/{strategy}/repeat-{repeat}/",
                    "test_enabled": False,
                })
    return {
        "schema_version": 1,
        "trials_per_format": trials_per_format,
        "route_count": len(routes),
        "routes": routes,
    }


def _validate_calibration_plan(plan: Any, trials_per_format: int) -> Dict[str, Any]:
    expected = _calibration_plan(trials_per_format)
    if plan != expected:
        raise ContractError(
            "calibration plan must be the exact canonical 8 x 7 x 2 stratified plan"
        )
    routes = plan["routes"]
    if len(routes) != CALIBRATION_ROUTE_COUNT:
        raise ContractError("calibration plan does not contain exactly 112 routes")
    run_keys = {route["run_key"] for route in routes}
    prefixes = {route["staging_prefix"] for route in routes}
    if len(run_keys) != len(routes) or len(prefixes) != len(routes):
        raise ContractError("calibration plan run keys and staging prefixes must be unique")
    if any(route.get("test_enabled") is not False or "test_pair_ids" in route for route in routes):
        raise ContractError("calibration routes must disable tests and omit test pair IDs")
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


def _manifest(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ContractError("preflight manifest must use schema_version 1")
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
    if not isinstance(layers, list) or not layers or any(type(layer) is not int for layer in layers):
        raise ContractError("preflight manifest layers must be a non-empty integer list")
    split = data.get("split", {})
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
    if normalized["train"] & normalized["validation"]:
        raise ContractError("train and validation supports must be disjoint")
    if split.get("selection_split") != "validation" or split.get("final_fit") != ["train"]:
        raise ContractError("calibration must fit on train and select on validation only")
    if split.get("test_evaluations") != 0:
        raise ContractError("calibration manifest must disable final test evaluation")
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


def _search_space(method: str, layers: Sequence[int], strategy: str):
    from wisent.core.utils.cli.commands.optimize_steering.pipeline.search_space import get_method_space
    from wisent.core.utils.services.optimization.core.parameters import CategoricalParam
    if strategy not in STRATEGIES:
        raise ContractError(f"unknown fixed extraction strategy {strategy!r}")
    space = get_method_space(method, max(layers) + 1)
    space["extraction_strategy"] = CategoricalParam(choices=[strategy])
    layer_key = "sensor_layer" if method.upper() in {"TETNO", "GROM"} else "layer"
    space[layer_key] = CategoricalParam(choices=list(layers))
    return space


def _run_hpo(args: argparse.Namespace, manifest_path: Path, manifest: Mapping[str, Any], routes) -> Path:
    method = manifest["job_unit"]["method"]
    if method == "baseline":
        raise ContractError("baseline has no HPO mode")
    destination = _destination(args, manifest, "hpo")
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
        for strategy_index, strategy in enumerate(STRATEGIES):
            work_dir = staging / "trial_work" / strategy
            work_dir.mkdir(parents=True)
            objective = create_objective(
                method=method, model=MODEL, task=BENCHMARK,
                num_layers=max(layers) + 1,
                limit=None, device=args.device, work_dir=str(work_dir),
                test_pairs_file=str(validation_file), strict_enriched_files=enriched,
                cached_model=cached_model,
            )
            result = optimizer.optimize_fn(
                objective, _search_space(method, layers, strategy),
                args.trials_per_format,
                cfg=HPOConfig(
                    backend="optuna", n_trials=args.trials_per_format,
                    seed=args.seed + strategy_index, sampler="tpe", pruner="nop",
                    load_if_exists=False,
                ),
                extra_trials=0,
            )
            if len(result.all_trials) != args.trials_per_format:
                raise ContractError(
                    f"format {strategy!r} completed {len(result.all_trials)} trials; "
                    f"expected {args.trials_per_format}"
                )
            best_params = dict(result.best_params)
            best_params["extraction_strategy"] = strategy
            scores = [trial.get("score", trial.get("value")) for trial in result.all_trials]
            format_results.append({
                "extraction_strategy": strategy,
                "trial_count": len(result.all_trials),
                "scores": scores,
                "best_params": best_params,
                "best_validation_score": result.best_score,
                "trials": result.all_trials,
            })
        if len(format_results) != len(STRATEGIES) or {
            item["trial_count"] for item in format_results
        } != {TRIALS_PER_FORMAT}:
            raise ContractError("HPO did not preserve the equal per-format trial budget")
        global_best = max(format_results, key=lambda item: item["best_validation_score"])
        global_params = dict(global_best["best_params"])
        identity = {
            "schema_version": 1,
            "mode": "hpo",
            "model": MODEL,
            "benchmark": BENCHMARK,
            "method": method,
            "manifest_sha256": _digest(manifest_path),
            "completion_index_sha256": _digest(args.completion_index.resolve()),
            "calibration_plan_sha256": _digest(args.calibration_plan.resolve()),
            "fit_pair_ids": train_ids,
            "selection_pair_ids": validation_ids,
            "test_pair_ids_read": [],
            "trials_per_format": args.trials_per_format,
            "trial_count": len(STRATEGIES) * args.trials_per_format,
            "per_format": format_results,
            "best_params": global_params,
            "best_validation_score": global_best["best_validation_score"],
        }
        _atomic_json(staging / "best_config.json", global_params)
        _atomic_json(staging / "validation_summary.json", {
            "selection_split": "validation", "pair_ids": validation_ids,
            "trials_per_format": args.trials_per_format,
            "trial_count": len(STRATEGIES) * args.trials_per_format,
            "per_format": [{
                "extraction_strategy": item["extraction_strategy"],
                "trial_count": item["trial_count"],
                "scores": item["scores"],
                "best_params": item["best_params"],
                "best_validation_score": item["best_validation_score"],
            } for item in format_results],
            "best_score": global_best["best_validation_score"],
            "best_params": global_params,
        })
        _atomic_json(staging / "trials.json", {"per_format": format_results})
        _atomic_json(staging / "frozen_config.json", identity)
        os.chmod(staging / "frozen_config.json", 0o444)
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
        _validate_calibration_plan(_read_json(args.calibration_plan), args.trials_per_format)
        destination = _run_hpo(args, manifest_path, manifest, routes)
        print(destination)
        return 0
    except (ContractError, ValueError, OSError) as exc:
        print(f"desired-results runner refused execution: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
