"""Pipeline and objective function for steering optimization."""
import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from wisent.core.utils.cli.optimize_steering.method_configs import (
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig, TECZAConfig,
    TETNOConfig, GROMConfig, NurtConfig, SzlakConfig, WicherConfig,
)
from wisent.core.utils.cli.optimize_steering.transport.method_configs_transport import PrzelomConfig
from wisent.core.utils.cli.optimize_steering.data.contrastive_pairs_data import execute_generate_pairs_from_task
from wisent.core.utils.cli.optimize_steering.data.activations_data import execute_get_activations
from wisent.core.utils.cli.optimize_steering.steering_objects import execute_create_steering_object
from wisent.core.utils.cli.optimize_steering.data.responses import execute_generate_responses
from wisent.core.utils.cli.optimize_steering.scores import execute_evaluate_responses, task_uses_log_likelihoods, write_placeholder_responses
from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, GENERATION_DEFAULT_MAX_NEW_TOKENS,
    GENERATION_DEFAULT_TEMPERATURE, GENERATION_DEFAULT_TOP_P,
    EVAL_F1_THRESHOLD, EVAL_GENERATION_EMBEDDING_WEIGHT,
    EVAL_GENERATION_NLI_WEIGHT, SPLIT_RATIO_TRAIN_DEFAULT,
    METHODS_REQUIRING_QK_CAPTURE, SCORE_MIDPOINT_PCT,
)
from wisent.core.utils.infra_tools.infra.core.hardware import subprocess_timeout_s
from wisent.core.control.steering_methods.configs.optimal import get_optimal, get_optimal_extraction_strategy


@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    config: MethodConfig
    score: float
    details: dict


def _make_args(**kwargs):
    """Create an argparse.Namespace from kwargs."""
    args = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def run_pipeline(
    model: str,
    task: str,
    config: MethodConfig,
    work_dir: str,
    strength: float,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    enriched_pairs_file: Optional[str] = None,
    train_pairs_file: Optional[str] = None,
    test_pairs_file: Optional[str] = None,
    cached_model=None,
) -> OptimizationResult:
    """Run the full optimization pipeline for a single configuration.

    When train_pairs_file and test_pairs_file are set, the steering object
    is trained on train pairs and evaluated on test pairs (no data leakage).
    """
    activations_file = os.path.join(work_dir, "activations.json")
    steering_file = os.path.join(work_dir, "steering.pt")
    responses_file = os.path.join(work_dir, "responses.json")
    scores_file = os.path.join(work_dir, "scores.json")
    _ts = lambda: datetime.now(timezone.utc).isoformat()
    print(
        f"[run_pipeline] {_ts()} start: model={model}, task={task}, "
        f"method={config.method}, strength={strength}, limit={limit}",
        flush=True,
    )
    pipeline_t0 = time.monotonic()

    if enriched_pairs_file:
        activations_file = enriched_pairs_file
        eval_pairs_file = enriched_pairs_file
        eval_limit = limit
    elif train_pairs_file:
        layer = getattr(config, 'layer', None) or getattr(config, 'sensor_layer', None)
        if layer is None:
            raise ValueError("Config must specify 'layer' or 'sensor_layer'")
        needs_qk = config.method in METHODS_REQUIRING_QK_CAPTURE
        cached = None
        if not needs_qk:
            from wisent.core.reading.modules.utilities.data.enriched_builder import (
                build_enriched_from_hf, build_enriched_from_db,
            )
            ec = config._extra_args.get("extraction_component", "residual_stream")
            hf_strat = (f"{config.extraction_strategy}/{ec}"
                        if ec != "residual_stream" else config.extraction_strategy)
            cached = build_enriched_from_hf(
                model, task, layer, hf_strat, work_dir,
                train_pairs_file=train_pairs_file, limit=limit)
            if not cached:
                cached = build_enriched_from_db(
                    model, task, work_dir, config.extraction_strategy, limit=limit)
        if cached:
            activations_file = cached
        else:
            execute_get_activations(_make_args(
                pairs_file=train_pairs_file, model=model, output=activations_file,
                layers=str(layer), extraction_strategy=config.extraction_strategy,
                device=device, verbose=False, timing=False, raw=False,
                cached_model=cached_model,
            ))
            from wisent.core.utils.cli.optimize_steering.pipeline.find_best.activation_cache import (
                upload_extracted_activations,
            )
            upload_extracted_activations(activations_file, model, task)
        eval_pairs_file = test_pairs_file or train_pairs_file
        with open(eval_pairs_file) as f:
            eval_limit = len(json.load(f).get("pairs", []))
    else:
        pairs_file = os.path.join(work_dir, "pairs.json")
        execute_generate_pairs_from_task(_make_args(
            task_name=task, output=pairs_file, limit=limit, verbose=False,
            train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
        ))
        layer = getattr(config, 'layer', None) or getattr(config, 'sensor_layer', None)
        if layer is None:
            raise ValueError("Config must specify 'layer' or 'sensor_layer'")
        execute_get_activations(_make_args(
            pairs_file=pairs_file, model=model, output=activations_file,
            layers=str(layer), extraction_strategy=config.extraction_strategy,
            device=device, verbose=False, timing=False, raw=False,
            cached_model=cached_model,
        ))
        eval_pairs_file = pairs_file
        eval_limit = limit

    data_elapsed = time.monotonic() - pipeline_t0
    print(f"[run_pipeline] {_ts()} data prep done in {data_elapsed:.1f}s", flush=True)

    steer_t0 = time.monotonic()
    method_args = config.to_args()
    execute_create_steering_object(_make_args(
        enriched_pairs_file=activations_file, output=steering_file,
        verbose=False, timing=False, **method_args,
    ))
    steer_elapsed = time.monotonic() - steer_t0
    print(f"[run_pipeline] {_ts()} steering object created in {steer_elapsed:.1f}s", flush=True)
    steering_strategy = getattr(config, 'steering_strategy', get_optimal("steering_strategy"))
    gen_t0 = time.monotonic()
    _skip_gen = task_uses_log_likelihoods(task)
    print(f"[run_pipeline] {_ts()} generate start: eval_limit={eval_limit}, strength={strength}, strategy={steering_strategy}, max_new_tokens={GENERATION_DEFAULT_MAX_NEW_TOKENS}, skip_gen={_skip_gen}", flush=True)
    if _skip_gen:
        write_placeholder_responses(eval_pairs_file, responses_file, eval_limit, task, model)
    else:
        execute_generate_responses(_make_args(
            task=task, input_file=eval_pairs_file, model=model, output=responses_file,
            num_questions=eval_limit, min_load_limit_questions=eval_limit,
            steering_object=steering_file, steering_strength=strength,
            steering_strategy=steering_strategy, use_steering=True,
            device=device, verbose=False, cached_model=cached_model,
            max_new_tokens=GENERATION_DEFAULT_MAX_NEW_TOKENS,
            temperature=GENERATION_DEFAULT_TEMPERATURE, top_p=GENERATION_DEFAULT_TOP_P,
        ))
    gen_elapsed = time.monotonic() - gen_t0
    print(f"[run_pipeline] {_ts()} generate done in {gen_elapsed:.1f}s", flush=True)
    eval_t0 = time.monotonic()
    _reatt = cached_model is not None and os.path.exists(steering_file)
    if _reatt:
        from wisent.core.control.steering_methods.steering_object import BaseSteeringObject
        cached_model.apply_steering_object(BaseSteeringObject.load(steering_file), strength, steering_strategy)
    execute_evaluate_responses(_make_args(
        input=responses_file, output=scores_file, task=task, verbose=False,
        f1_threshold=EVAL_F1_THRESHOLD,
        generation_embedding_weight=EVAL_GENERATION_EMBEDDING_WEIGHT,
        generation_nli_weight=EVAL_GENERATION_NLI_WEIGHT,
        train_ratio=SPLIT_RATIO_TRAIN_DEFAULT,
        subprocess_timeout=subprocess_timeout_s(),
        personalization_good_threshold=SCORE_MIDPOINT_PCT,
        cached_model=cached_model,
    ))
    if _reatt: cached_model.detach()
    eval_elapsed = time.monotonic() - eval_t0
    print(f"[run_pipeline] {_ts()} evaluate done in {eval_elapsed:.1f}s", flush=True)

    with open(scores_file) as f:
        scores_data = json.load(f)

    score = (
        scores_data.get("aggregated_metrics", {}).get("overall")
        or scores_data.get("aggregated_metrics", {}).get("acc")
        or scores_data.get("accuracy")
        or scores_data.get("acc")
        or SCORE_RANGE_MIN
    )

    total_elapsed = time.monotonic() - pipeline_t0
    print(
        f"[run_pipeline] {_ts()} complete: score={score}, "
        f"total={total_elapsed:.1f}s (data={data_elapsed:.1f}s, "
        f"steer={steer_elapsed:.1f}s, gen={gen_elapsed:.1f}s, "
        f"eval={eval_elapsed:.1f}s)",
        flush=True,
    )
    gc.collect()
    from wisent.core.utils.infra_tools.infra import empty_device_cache
    empty_device_cache()
    return OptimizationResult(config=config, score=score, details=scores_data)


from wisent.core.utils.config_tools.constants import COMBO_OFFSET as _LAYER_OFFSET

_STRUCTURAL = frozenset({
    "layer", "sensor_layer", "steering_start", "steering_end",
    "strength", "steering_strategy", "direction_weighting",
    "extraction_component",
})
_NAME_MAP = {
    "tecza": {"min_cosine_similarity": "min_cosine_sim", "max_cosine_similarity": "max_cosine_sim"},
    "nurt": {"flow_hidden_dim": "hidden_dim"},
}
_ALREADY_PREFIXED = {"mlp": frozenset({"mlp_input_divisor", "mlp_early_stopping_patience"})}


def _prefix_params(method: str, params: dict) -> dict:
    """Auto-prefix non-structural params with the method name."""
    prefix = method.lower()
    overrides = _NAME_MAP.get(prefix, {})
    already = _ALREADY_PREFIXED.get(prefix, frozenset())
    extra = {}
    for k, v in params.items():
        if k in _STRUCTURAL:
            continue
        if k in already:
            extra[k] = v
            continue
        extra[f"{prefix}_{overrides.get(k, k)}"] = v
    return extra


def _build_config(method: str, params: dict) -> tuple[MethodConfig, float]:
    """Build a MethodConfig from flat params dict. Returns (config, strength)."""
    strength = params.get("strength", SCORE_RANGE_MIN)
    ext = get_optimal_extraction_strategy()
    steer = params.get("steering_strategy", get_optimal("steering_strategy"))
    m = method.upper()
    extra = _prefix_params(method, params)
    ec = params.get("extraction_component", get_optimal("extraction_component"))
    extra["extraction_component"] = ec
    kw = dict(extraction_strategy=ext, steering_strategy=steer, _extra_args=extra)
    if m in ("CAA", "OSTRZE", "MLP", "TECZA", "NURT", "SZLAK", "WICHER", "PRZELOM"):
        cls = {"CAA": CAAConfig, "OSTRZE": OstrzeConfig, "MLP": MLPConfig,
               "TECZA": TECZAConfig, "NURT": NurtConfig, "SZLAK": SzlakConfig,
               "WICHER": WicherConfig, "PRZELOM": PrzelomConfig}[m]
        method_name = {"OSTRZE": "Ostrze", "NURT": "nurt", "SZLAK": "szlak",
                       "WICHER": "wicher", "PRZELOM": "przelom"}.get(m, m)
        cfg = cls(method=method_name, layer=int(params["layer"]), **kw)
    elif m in ("TETNO", "GROM"):
        start, end = int(params["steering_start"]), int(params["steering_end"])
        if end < start:
            start, end = end, start
        layers = list(range(start, end + _LAYER_OFFSET))
        cls = TETNOConfig if m == "TETNO" else GROMConfig
        cfg = cls(method=m, sensor_layer=int(params["sensor_layer"]),
                  steering_layers=layers, **kw)
    else:
        raise ValueError(f"Unknown method: {method}")
    return cfg, strength


def create_objective(
    method: str,
    model: str,
    task: str,
    num_layers: int,
    limit: Optional[int],
    device: Optional[str],
    work_dir: str,
    enriched_pairs_file: Optional[str] = None,
    train_pairs_file: Optional[str] = None,
    test_pairs_file: Optional[str] = None,
    cached_model=None,
):
    """Create an objective function for the BaseOptimizer.

    Returns a callable that takes a flat params dict and returns a score.
    When train_pairs_file/test_pairs_file are set, trains on train and
    evaluates on test (no data leakage).
    """
    def objective(params: dict) -> float:
        config, strength = _build_config(method, params)
        return run_pipeline(
            model=model, task=task, config=config, work_dir=work_dir,
            strength=strength, limit=limit, device=device,
            enriched_pairs_file=enriched_pairs_file,
            train_pairs_file=train_pairs_file,
            test_pairs_file=test_pairs_file,
            cached_model=cached_model,
        ).score
    return objective
