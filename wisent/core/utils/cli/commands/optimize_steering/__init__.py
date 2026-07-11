import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""
Steering optimization using the wisent CLI pipeline.

Pipeline:
1. contrastive_pairs  -> wisent generate-pairs-from-task
2. activations        -> wisent get-activations
3. steering_objects   -> wisent create-steering-object
4. responses          -> wisent generate-responses
5. scores             -> wisent evaluate-responses

Search space per method:
- CAA/Ostrze: layer, extraction_strategy
- MLP: layer, extraction_strategy, hidden_dim, num_layers
- TECZA: layer, extraction_strategy, num_directions, direction_weighting, retain_weight
- TETNO: sensor_layer, steering_layers, condition_threshold, gate_temperature, max_alpha
- GROM: sensor_layer, steering_layers, num_directions, gate_hidden_dim, intensity_hidden_dim, behavior_weight

WARNING: NEVER REDUCE THE SEARCH SPACE
- Do NOT discretize continuous parameters into a few values
- Do NOT limit integer ranges to small sets like [1,2,3]
- If a parameter is continuous (0.0-1.0), keep it continuous
- If a parameter can be any integer (e.g. layer 0-27), search ALL values
- Let the sampling strategy (random, Bayesian, etc.) decide which points to evaluate
- Any reduction in search space requires EXPLICIT user approval
"""

import json
import tempfile
from typing import Optional

# Re-exports for backwards compatibility
from wisent.core.utils.cli.optimize_steering.method_configs import (
    STEERING_STRATEGIES,
    MethodConfig, CAAConfig, OstrzeConfig, MLPConfig,
    TECZAConfig, TETNOConfig, GROMConfig, NurtConfig,
    SzlakConfig, WicherConfig,
)
from wisent.core.utils.cli.optimize_steering.transport.method_configs_transport import PrzelomConfig
from wisent.core.utils.cli.optimize_steering.search_space import get_method_space
from wisent.core.utils.services.optimization.core.atoms import BaseOptimizer, HPOConfig
from wisent.core.utils.config_tools.constants import JSON_INDENT, SEPARATOR_WIDTH_REPORT, TRIALS_PER_DIMENSION_MULTIPLIER

_LAZY_EXPORTS = {
    "OptimizationResult": ("wisent.core.utils.cli.optimize_steering.pipeline", "OptimizationResult"),
    "run_pipeline": ("wisent.core.utils.cli.optimize_steering.pipeline", "run_pipeline"),
    "create_objective": ("wisent.core.utils.cli.optimize_steering.pipeline", "create_objective"),
    "_make_args": ("wisent.core.utils.cli.optimize_steering.pipeline", "_make_args"),
    "execute_transport_rl": ("wisent.core.utils.cli.optimize_steering.transport.transport_rl", "execute_transport_rl"),
    "execute_continual_learning": ("wisent.core.utils.cli.optimize_steering.continual", "execute_continual_learning"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def execute_optimize_steering(args):
    """
    Execute steering optimization.

    Routes to the appropriate handler based on subcommand:
    - auto: Uses zwiad geometry analysis to select method, then grid search
    - welfare: Welfare state steering optimization (ANIMA framework)
    - personalization: Personality trait steering optimization
    - Other subcommands: Use Optuna-based optimization
    """
    # Check for 'auto' subcommand - route to zwiad-based optimizer
    subcommand = getattr(args, 'subcommand', None)
    if subcommand == 'auto':
        from wisent.core.control.steering_optimizer import run_auto_steering_optimization

        _GROM_PARAM_NAMES = [
            "num_directions", "optimization_steps", "learning_rate",
            "warmup_steps", "behavior_weight", "retain_weight",
            "sparse_weight", "smooth_weight", "independence_weight",
            "max_alpha", "gate_temperature", "max_grad_norm",
            "eta_min_factor", "linear_threshold",
            "adapt_cone_threshold", "adapt_manifold_threshold",
            "adapt_linear_directions", "adapt_complex_directions",
            "adapt_max_directions", "significant_directions_default",
            "min_adapted_directions", "caa_similarity_skip",
            "contrastive_margin", "contrastive_weight", "utility_weight",
            "concentration_weight", "gate_warmup_weight", "caa_alignment_weight",
            "gate_dim_min", "gate_dim_max", "gate_dim_divisor",
            "intensity_dim_min", "intensity_dim_max", "intensity_dim_divisor",
            "log_interval", "weight_decay",
        ]
        _TETNO_PARAM_NAMES = [
            "condition_threshold", "gate_temperature", "max_alpha",
            "entropy_floor", "entropy_ceiling", "threshold_search_steps",
            "learning_rate", "condition_margin", "min_layer_scale", "log_interval",
            "optimization_steps",
        ]
        grom_params = {k: getattr(args, f"grom_{k}") for k in _GROM_PARAM_NAMES}
        tetno_params = {k: getattr(args, f"tetno_{k}") for k in _TETNO_PARAM_NAMES}
        result = run_auto_steering_optimization(
            model_name=args.model,
            task_name=args.task,
            limit=None,
            min_norm_threshold=args.min_norm_threshold,
            device=getattr(args, 'device', None),
            verbose=getattr(args, 'verbose', False),
            max_time_minutes=args.max_time,
            methods_to_test=args.methods,
            layer_range=getattr(args, 'layer_range', None),
            strength_range=getattr(args, 'strength_range', None),
            min_clusters=getattr(args, 'min_clusters', None),
            grom_params=grom_params,
            tetno_params=tetno_params,
            tecza_params={
                "tecza_num_directions": args.tecza_num_directions,
                "tecza_learning_rate": args.tecza_learning_rate,
                "tecza_retain_weight": args.tecza_retain_weight,
                "tecza_independence_weight": args.tecza_independence_weight,
                "tecza_min_cosine_sim": args.tecza_min_cosine_similarity,
                "tecza_max_cosine_sim": args.tecza_max_cosine_similarity,
                "tecza_marginal_threshold": args.tecza_marginal_threshold,
                "tecza_max_directions": args.tecza_max_directions,
                "tecza_ablation_weight": args.tecza_ablation_weight,
                "tecza_addition_weight": args.tecza_addition_weight,
                "tecza_separation_margin": args.tecza_separation_margin,
                "tecza_perturbation_scale": args.tecza_perturbation_scale,
                "tecza_universal_basis_noise": args.tecza_universal_basis_noise,
                "tecza_optimization_steps": args.tecza_optimization_steps,
            },
            auto_min_pairs=args.auto_min_pairs,
            auto_sample_size=args.auto_sample_size,
            auto_n_folds=args.auto_n_folds,
            auto_min_pairs_split=args.auto_min_pairs_split,
            auto_layer_divisor=args.auto_layer_divisor,
            architecture_module_limit=args.architecture_module_limit,
            progress_log_interval=args.progress_log_interval,
            train_ratio=args.train_ratio,
            probe_small_hidden=args.probe_small_hidden,
            probe_mlp_hidden=args.probe_mlp_hidden,
            probe_mlp_alpha=args.probe_mlp_alpha,
            spectral_n_neighbors=args.spectral_n_neighbors,
            direction_n_bootstrap=args.direction_n_bootstrap,
            direction_subset_fraction=args.direction_subset_fraction,
            direction_std_penalty=args.direction_std_penalty,
            consistency_w_cosine=args.consistency_w_cosine,
            consistency_w_positive=args.consistency_w_positive,
            consistency_w_high_sim=args.consistency_w_high_sim,
            sparsity_threshold_fraction=args.sparsity_threshold_fraction,
            detection_threshold=args.detection_threshold,
            direction_moderate_similarity=args.direction_moderate_similarity,
        )

        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return None

        return result

    # Check for 'comprehensive' steering_action - multi-method comparison
    steering_action = getattr(args, 'steering_action', None)
    if steering_action == 'comprehensive':
        from .pipeline.comprehensive import execute_comprehensive_optimization
        return execute_comprehensive_optimization(args)

    # Check for 'welfare' steering_action - AI welfare states optimization
    if steering_action == 'welfare':
        from wisent.core.utils.cli.optimize_steering.welfare import _execute_welfare_optimization
        return _execute_welfare_optimization(args)

    # Check for 'personalization' steering_action
    if steering_action == 'personalization':
        from wisent.core.utils.cli.optimize_steering.personalization import _execute_personalization_optimization
        return _execute_personalization_optimization(args)

    # Check for 'hierarchical' steering_action
    if steering_action == 'hierarchical':
        from .hierarchical import execute_hierarchical_optimization
        return execute_hierarchical_optimization(args)

    # Check for 'transport-rl' steering_action
    if steering_action == 'transport-rl':
        from wisent.core.utils.cli.optimize_steering.transport.transport_rl import execute_transport_rl
        return execute_transport_rl(args)

    # Check for 'continual' steering_action (subcommand)
    if steering_action == 'continual' or subcommand == 'continual':
        from wisent.core.utils.cli.optimize_steering.continual import execute_continual_learning
        return execute_continual_learning(args)

    # Default: Unified optimizer (Hyperopt or Optuna backend)
    method = args.method
    enriched_pairs_file = getattr(args, 'enriched_pairs_file', None)
    task = args.task
    backend = args.backend
    device = getattr(args, 'device', None)

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"STEERING OPTIMIZATION ({backend})")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"   Model: {args.model}")
    if enriched_pairs_file:
        print(f"   Data: {enriched_pairs_file}")
    else:
        print(f"   Task: {task}")
    print(f"   Method: {method}")
    print(f"   Backend: {backend}")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}\n")

    from transformers import AutoConfig as _AC
    _cfg = _AC.from_pretrained(args.model, trust_remote_code=True)
    num_layers = _cfg.num_hidden_layers

    if enriched_pairs_file:
        with open(enriched_pairs_file) as f:
            data = json.load(f)
        num_layers = len(data.get("layers", [])) or num_layers
        print(f"   Loaded {num_layers} layers from enriched pairs file")

    space = get_method_space(method, num_layers)
    n_trials = len(space) * TRIALS_PER_DIMENSION_MULTIPLIER
    optimizer = BaseOptimizer()
    optimizer.direction = "maximize"

    from wisent.core.utils.cli.optimize_steering.pipeline import create_objective

    with tempfile.TemporaryDirectory() as work_dir:
        objective = create_objective(
            method=method, model=args.model, task=task,
            num_layers=num_layers, limit=None, device=device,
            work_dir=work_dir, enriched_pairs_file=enriched_pairs_file,
        )
        result = optimizer.optimize_fn(objective, space, n_trials, cfg=HPOConfig(backend=backend))

    print(f"\n{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'=' * SEPARATOR_WIDTH_REPORT}")
    print(f"\n   Best configuration:")
    print(f"   Method: {method}")
    print(f"   Score: {result.best_score:.4f}")
    for k, v in result.best_params.items():
        print(f"   {k}: {v}")

    # Replay best config to persist responses + scores
    replay_dir = getattr(args, 'output_dir', None) or './optimization_results'
    import os
    os.makedirs(replay_dir, exist_ok=True)
    from wisent.core.utils.cli.optimize_steering.pipeline import (
        run_pipeline, _build_config,
    )
    config, strength = _build_config(method, result.best_params)
    print(f"\n   Replaying best config to persist responses...")
    replay_result = run_pipeline(
        model=args.model, task=task, config=config,
        work_dir=replay_dir, strength=strength,
        limit=None, device=device,
        enriched_pairs_file=enriched_pairs_file,
    )
    print(f"   Responses saved to: {replay_dir}/responses.json")
    print(f"   Scores saved to: {replay_dir}/scores.json")

    output_data = {
        "model": args.model,
        "task": task,
        "method": method,
        "n_trials": n_trials,
        "backend": backend,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "all_trials": result.all_trials,
    }
    summary_path = os.path.join(replay_dir, "optimization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)

    if hasattr(args, 'output') and args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=JSON_INDENT)
        print(f"\n   Results saved to: {args.output}")

    return result


__all__ = [
    "execute_optimize_steering",
    "run_pipeline",
    "get_method_space",
    "create_objective",
    "BaseOptimizer",
    "MethodConfig",
    "CAAConfig",
    "OstrzeConfig",
    "MLPConfig",
    "TECZAConfig",
    "TETNOConfig",
    "GROMConfig",
    "NurtConfig",
    "SzlakConfig",
    "WicherConfig",
    "PrzelomConfig",
    "OptimizationResult",
    "execute_transport_rl",
    "execute_continual_learning",
]
