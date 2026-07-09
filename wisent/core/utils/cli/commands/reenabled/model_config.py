"""CLI execution for the re-enabled 'model-config' command.

Dispatches on ``args.config_action`` in {save, list, show, remove, test},
mirroring the switch + print(json) style of ``execute_inference_config``
(wisent/core/utils/cli/analysis/analysis/config/inference_config_cli.py).

All heavy dependencies (torch / transformers / WisentModel) are imported
lazily inside the functions that need them; the config-management primitives
are also imported lazily so importing this module stays cheap.
"""

import json

from wisent.core.utils.config_tools.constants import JSON_INDENT


# Defaults for the classification / steering config fields that the
# `model-config save` parser does NOT collect
# (.../parser_arguments/core/configuration/model_config_parser.py only exposes:
# model, classification-layer, steering-layer, detection-threshold,
# optimization-method, metrics). The underlying save primitives require the
# fields below, so we fill them with values drawn from the codebase's own
# choice lists and empirically-optimal parameter values:
#   aggregation/targeting choices -> generate_vector_parser.py
#   prompt_construction / prompt_strategy / normalize_mode / steering_strategy /
#   direction_weighting -> steering_methods/configs/parameters_to_validate.json
_DEFAULT_TOKEN_AGGREGATION = "last_token"
_DEFAULT_TOKEN_TARGETING = "last_token"
_DEFAULT_CLASSIFIER_TYPE = "logistic"
_DEFAULT_PROMPT_CONSTRUCTION = "chat_template"
_DEFAULT_STEERING_METHOD = "CAA"
_DEFAULT_PROMPT_STRATEGY = "default"
_DEFAULT_NORMALIZE_MODE = "l2"
_DEFAULT_STEERING_STRATEGY = "constant"
_DEFAULT_STEERING_METRIC = "accuracy"
_DEFAULT_DIRECTION_WEIGHTING = "primary_only"

# Keys accepted in --metrics JSON that map onto real ClassificationConfig
# quality fields (manager/classification.py). Other keys are echoed in the
# printed result but are not persisted (the config schema has no generic
# metrics field: ModelConfig in config/types.py).
_METRIC_FIELD_MAP = {
    "accuracy": "accuracy",
    "f1": "f1_score",
    "f1_score": "f1_score",
    "precision": "precision",
    "recall": "recall",
}


def _build_manager(config_dir):
    """Build the ModelConfigManager primitive, honoring --config-dir."""
    # Documented primitive (backward_compat/model_config_manager.py -> class ModelConfigManager).
    from wisent.core.utils.config_tools.config.backward_compat.model_config_manager import (
        ModelConfigManager,
    )

    mgr = ModelConfigManager(config_dir=config_dir)  # ctor accepts config_dir (model_config_manager.py __init__)
    if config_dir:
        # The backward-compat ctor DISCARDS config_dir (its __init__ binds the process-global
        # singleton via get_config_manager()). Rebind the underlying manager so --config-dir is
        # honored (WisentConfigManagerBase.__init__ expands & creates config_dir -> manager/base.py).
        from wisent.core.utils.config_tools.config.manager import WisentConfigManager  # manager/__init__.py

        mgr._manager = WisentConfigManager(config_dir=config_dir)
    return mgr


def _parse_metrics(raw):
    """Parse the --metrics JSON string into a dict (empty when unset)."""
    if not raw:
        return {}
    parsed = json.loads(raw)  # raises JSONDecodeError on malformed input -> reported by caller
    if not isinstance(parsed, dict):
        raise ValueError("--metrics must be a JSON object")
    return parsed


def _save(mgr, args):
    """save: persist classification + steering config built from CLI args."""
    metrics = _parse_metrics(getattr(args, "metrics", None))
    quality = {
        dst: metrics[src]
        for src, dst in _METRIC_FIELD_MAP.items()
        if metrics.get(src) is not None
    }
    steering_layer = args.steering_layer if args.steering_layer is not None else args.classification_layer

    # NOTE: the primitive ModelConfigManager.save_model_config is currently broken -- its body
    # forwards only a few args to save_steering_config, omitting the six positional args that
    # SteeringMixin.save_steering_config now requires (manager/steering.py), so it raises
    # TypeError for every model. We therefore drive the underlying, working primitives directly
    # -- exactly what save_model_config intended.
    mgr._manager.save_classification_config(  # manager/classification.py save_classification_config
        model_name=args.model,
        layer=args.classification_layer,
        token_aggregation=_DEFAULT_TOKEN_AGGREGATION,
        classifier_type=_DEFAULT_CLASSIFIER_TYPE,
        token_targeting_strategy=_DEFAULT_TOKEN_TARGETING,
        prompt_construction_strategy=_DEFAULT_PROMPT_CONSTRUCTION,
        detection_threshold=args.detection_threshold,
        optimization_method=args.optimization_method,
        set_as_default=True,
        **quality,
    )
    mgr._manager.save_steering_config(  # manager/steering.py save_steering_config
        model_name=args.model,
        method=_DEFAULT_STEERING_METHOD,
        token_aggregation=_DEFAULT_TOKEN_AGGREGATION,
        prompt_strategy=_DEFAULT_PROMPT_STRATEGY,
        normalize_mode=_DEFAULT_NORMALIZE_MODE,
        strategy=_DEFAULT_STEERING_STRATEGY,
        optimization_method=args.optimization_method,
        metric=_DEFAULT_STEERING_METRIC,
        direction_weighting=_DEFAULT_DIRECTION_WEIGHTING,
        layer=steering_layer,
        set_as_default=True,
    )
    config_path = mgr._manager._get_config_path(args.model)  # manager/base.py _get_config_path

    unpersisted = sorted(k for k in metrics if k not in _METRIC_FIELD_MAP)
    return {
        "status": "saved",
        "model": args.model,
        "config_path": str(config_path),
        "classification_layer": args.classification_layer,
        "steering_layer": steering_layer,
        "detection_threshold": args.detection_threshold,
        "optimization_method": args.optimization_method,
        "metrics": metrics,
        "persisted_metric_fields": sorted(quality.keys()),
        "unpersisted_metric_keys": unpersisted,
    }


def _list(mgr, detailed):
    """list: enumerate saved model configs."""
    configs = mgr.list_model_configs()  # model_config_manager.py list_model_configs
    if detailed:
        configs = [mgr.load_model_config(c["model_name"]) for c in configs]  # load_model_config
    return {"status": "ok", "count": len(configs), "configs": configs}


def _show(mgr, model, task):
    """show: load and print one model's config (+ task overrides if present)."""
    cfg = mgr.load_model_config(model)  # model_config_manager.py load_model_config
    if cfg is None:
        return {"status": "not_found", "model": model}
    result = {"status": "ok", "model": model, "config": cfg}
    if task:
        full = mgr._manager.get_model_config(model)  # manager/traits.py get_model_config
        task_cfg = full.tasks.get(task)  # ModelConfig.tasks (config/types.py)
        result["task"] = task
        result["task_overrides"] = task_cfg.to_dict() if task_cfg else None  # TaskConfig.to_dict
    return result


def _remove(mgr, args):
    """remove: delete a model config (requires --confirm)."""
    if not getattr(args, "confirm", False):
        return {
            "status": "aborted",
            "model": args.model,
            "message": "Refusing to remove without --confirm.",
        }
    if not mgr.has_model_config(args.model):  # model_config_manager.py has_model_config
        return {"status": "not_found", "model": args.model}
    removed = mgr.remove_model_config(args.model)  # model_config_manager.py remove_model_config
    return {"status": "removed" if removed else "not_found", "model": args.model}


def _test(mgr, args):
    """test: validate saved config, then load the model to check layers in range."""
    cfg = mgr.load_model_config(args.model)  # model_config_manager.py load_model_config
    if cfg is None:
        return {
            "status": "not_found",
            "model": args.model,
            "message": "No saved configuration; run 'model-config save' first.",
        }

    params = cfg.get("optimal_parameters", {})
    class_layer = params.get("classification_layer")
    steer_layer = params.get("steering_layer")
    threshold = params.get("detection_threshold")

    errors = []
    if not isinstance(class_layer, int):
        errors.append("classification_layer missing or not an int")
    if steer_layer is not None and not isinstance(steer_layer, int):
        errors.append("steering_layer present but not an int")
    if threshold is None:
        errors.append("detection_threshold missing")

    result = {
        "status": "ok" if not errors else "invalid_config",
        "model": args.model,
        "task": args.task,
        "validation": {
            "classification_layer": class_layer,
            "steering_layer": steer_layer,
            "detection_threshold": threshold,
            "errors": errors,
            "config_valid": not errors,
        },
    }
    if errors:
        return result

    # Full verification requires the actual model: load it and confirm the configured layers
    # index into the decoder stack. Lazy import so the heavy torch/transformers stack only
    # loads for this action.
    try:
        from wisent.core.primitives.models.core.wisent_model import WisentModel  # wisent_model.py class WisentModel

        wm = WisentModel(args.model, device=getattr(args, "device", None))  # wisent_model.py __init__
        num_layers = wm.num_layers  # wisent_model.py num_layers property
        # A valid decoder index is a member of range(num_layers); the membership test avoids
        # an inline numeric bound.
        layer_checks = {}
        for name, layer in (("classification_layer", class_layer), ("steering_layer", steer_layer)):
            if layer is None:
                continue
            layer_checks[name] = {"layer": layer, "in_range": layer in range(num_layers)}
        all_in_range = all(c["in_range"] for c in layer_checks.values())
        result["model_load"] = {
            "loaded": True,
            "device": wm.device,
            "num_layers": num_layers,
            "hidden_size": wm.hidden_size,  # wisent_model.py hidden_size property
            "layers": layer_checks,
            "all_layers_in_range": all_in_range,
        }
        result["status"] = "ok" if all_in_range else "layers_out_of_range"
    except Exception as exc:  # model / weights / env unavailable
        result["model_load"] = {
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
            "note": (
                "Config fields validated successfully. Full verification (layer-range "
                "check) requires loading the model weights via a working "
                "torch/transformers/accelerate environment."
            ),
        }
        result["status"] = "config_valid_model_unavailable"
    return result


def execute_model_config(args):
    """Execute the model-config command."""
    action = getattr(args, "config_action", None)
    config_dir = getattr(args, "config_dir", None)
    verbose = getattr(args, "verbose", False)

    try:
        mgr = _build_manager(config_dir)
        if action == "save":
            result = _save(mgr, args)
        elif action == "list":
            result = _list(mgr, getattr(args, "detailed", False))
        elif action == "show":
            result = _show(mgr, args.model, getattr(args, "task", None))
        elif action == "remove":
            result = _remove(mgr, args)
        elif action == "test":
            result = _test(mgr, args)
        elif action is None:
            result = {
                "status": "error",
                "message": "No action specified. Choose one of: save, list, show, remove, test.",
            }
        else:
            result = {"status": "error", "message": f"Unknown config_action: {action}"}
    except Exception as exc:
        result = {"status": "error", "action": action, "error": f"{type(exc).__name__}: {exc}"}
        if verbose:
            import traceback

            result["traceback"] = traceback.format_exc()

    print(json.dumps(result, indent=JSON_INDENT, default=str))
    return result
