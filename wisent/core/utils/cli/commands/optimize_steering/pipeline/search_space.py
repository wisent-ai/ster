"""Per-method search space definitions.

Parameters with no code-enforced bounds use uninformative lognormal/normal
priors (wide sigma). Parameters with code-enforced finite ranges use uniform.
"""
from __future__ import annotations

from wisent.core.utils.services.optimization.core.parameters import (
    CategoricalParam as _Cat,
    FloatParam as _Fp,
    IntParam as _Ip,
    Param,
)
from wisent.core.utils.config_tools import constants as _C
from wisent.core.control.steering_optimizer.types import SteeringApplicationStrategy
from wisent.core.primitives.model_interface.core.activations.pipeline.strategies.extraction_strategy import ExtractionComponent

_SS = [s.value for s in SteeringApplicationStrategy]
_EC = ExtractionComponent.list_all()
_ES = [
    "chat_first",
    "chat_last",
    "chat_mean",
    "chat_max_norm",
    "chat_weighted",
    "mc_balanced",
    "role_play",
]
_DW = ["primary_only", "equal"]
U_MU = _C.UNINFORMATIVE_MU
U_SIG = _C.UNINFORMATIVE_SIGMA


def _ln(mu, sigma):
    return _Fp(distribution="lognormal", mu=mu, sigma=sigma)


def _nm(mu, sigma):
    return _Fp(distribution="normal", mu=mu, sigma=sigma)


def _uf(lo, hi):
    return _Fp(distribution="uniform", low=lo, high=hi)


def _ri(lo, hi):
    return _Ip(distribution="randint", low=lo, high=hi)


def _qln(mu, sigma):
    return _Ip(distribution="qlognormal", mu=mu, sigma=sigma)


def _base(num_layers: int) -> dict[str, Param]:
    # Exclude the final layer — steering on it causes CUDA assert
    # because there is no residual stream after the last transformer block.
    max_layer = num_layers - 1
    return {
        "steering_strategy": _Cat(choices=_SS),
        "extraction_strategy": _Cat(choices=_ES),
        "strength": _ln(U_MU, U_SIG),
        "layer": _ri(_C.SS_LAYER_MIN, max_layer),
    }


def _deferred_base(num_layers: int) -> dict[str, Param]:
    """Legacy Q/K method space; intentionally retains component search."""
    max_layer = num_layers - 1
    return {
        "steering_strategy": _Cat(choices=_SS),
        "extraction_component": _Cat(choices=_EC),
        "strength": _ln(U_MU, U_SIG),
        "layer": _ri(_C.SS_LAYER_MIN, max_layer),
    }


def _sensor_base(num_layers: int) -> dict[str, Param]:
    max_layer = num_layers - 1
    return {
        "steering_strategy": _Cat(choices=_SS),
        "extraction_strategy": _Cat(choices=_ES),
        "strength": _ln(U_MU, U_SIG),
        "sensor_layer": _ri(_C.SS_LAYER_MIN, max_layer),
        "steering_start": _ri(_C.SS_LAYER_MIN, max_layer),
        "steering_end": _ri(_C.SS_LAYER_MIN, max_layer),
    }


def caa_space(num_layers: int) -> dict[str, Param]:
    return _base(num_layers)


def ostrze_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["C"] = _ln(U_MU, U_SIG)
    return s


def mlp_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["hidden_dim"] = _qln(U_MU, U_SIG)
    s["num_layers"] = _ri(_C.SS_MLP_NUM_LAYERS_MIN, _C.SS_MLP_NUM_LAYERS_MAX)
    s["dropout"] = _uf(_C.SS_MLP_DROPOUT_LOW, _C.SS_MLP_DROPOUT_HIGH)
    s["epochs"] = _qln(U_MU, U_SIG)
    s["learning_rate"] = _ln(U_MU, U_SIG)
    s["weight_decay"] = _ln(U_MU, U_SIG)
    s["early_stop_tol"] = _ln(U_MU, U_SIG)
    s["mlp_input_divisor"] = _ri(_C.SS_MLP_INPUT_DIVISOR_MIN, _C.SS_MLP_INPUT_DIVISOR_MAX)
    s["gating_hidden_dim_divisor"] = _ri(_C.SS_MLP_GATING_DIVISOR_MIN, _C.SS_MLP_GATING_DIVISOR_MAX)
    s["mlp_early_stopping_patience"] = _ri(_C.SS_MLP_PATIENCE_MIN, _C.SS_MLP_PATIENCE_MAX)
    return s


def tecza_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["num_directions"] = _qln(U_MU, U_SIG)
    s["optimization_steps"] = _qln(U_MU, U_SIG)
    s["retain_weight"] = _uf(_C.SS_TECZA_RETAIN_LOW, _C.SS_TECZA_RETAIN_HIGH)
    s["learning_rate"] = _ln(U_MU, U_SIG)
    s["independence_weight"] = _ln(U_MU, U_SIG)
    s["min_cosine_similarity"] = _uf(_C.SS_TECZA_MIN_COS_LOW, _C.SS_TECZA_MIN_COS_HIGH)
    s["max_cosine_similarity"] = _uf(_C.SS_TECZA_MAX_COS_LOW, _C.SS_TECZA_MAX_COS_HIGH)
    s["variance_threshold"] = _uf(_C.SS_TECZA_VAR_THRESH_LOW, _C.SS_TECZA_VAR_THRESH_HIGH)
    s["marginal_threshold"] = _uf(_C.SS_TECZA_MARG_THRESH_LOW, _C.SS_TECZA_MARG_THRESH_HIGH)
    s["max_directions"] = _ri(_C.SS_TECZA_MAX_DIR_MIN, _C.SS_TECZA_MAX_DIR_MAX)
    s["ablation_weight"] = _ln(U_MU, U_SIG)
    s["addition_weight"] = _ln(U_MU, U_SIG)
    s["separation_margin"] = _uf(_C.SS_TECZA_SEP_MARGIN_LOW, _C.SS_TECZA_SEP_MARGIN_HIGH)
    s["perturbation_scale"] = _ln(U_MU, U_SIG)
    s["universal_basis_noise"] = _ln(U_MU, U_SIG)
    s["log_interval"] = _ri(_C.SS_TECZA_LOG_INTERVAL_MIN, _C.SS_TECZA_LOG_INTERVAL_MAX)
    s["direction_weighting"] = _Cat(choices=_DW)
    return s


def tetno_space(num_layers: int) -> dict[str, Param]:
    s = _sensor_base(num_layers)
    s["condition_threshold"] = _uf(_C.SS_TETNO_COND_THRESH_LOW, _C.SS_TETNO_COND_THRESH_HIGH)
    s["gate_temperature"] = _ln(U_MU, U_SIG)
    s["entropy_floor"] = _uf(_C.SS_TETNO_ENT_FLOOR_LOW, _C.SS_TETNO_ENT_FLOOR_HIGH)
    s["entropy_ceiling"] = _uf(_C.SS_TETNO_ENT_CEIL_LOW, _C.SS_TETNO_ENT_CEIL_HIGH)
    s["max_alpha"] = _ln(U_MU, U_SIG)
    s["optimization_steps"] = _qln(U_MU, U_SIG)
    s["learning_rate"] = _ln(U_MU, U_SIG)
    s["threshold_search_steps"] = _ri(_C.SS_TETNO_THRESH_STEPS_MIN, _C.SS_TETNO_THRESH_STEPS_MAX)
    s["condition_margin"] = _uf(_C.SS_TETNO_COND_MARGIN_LOW, _C.SS_TETNO_COND_MARGIN_HIGH)
    s["min_layer_scale"] = _uf(_C.SS_TETNO_MIN_LAYER_SCALE_LOW, _C.SS_TETNO_MIN_LAYER_SCALE_HIGH)
    s["log_interval"] = _ri(_C.SS_TETNO_LOG_INTERVAL_MIN, _C.SS_TETNO_LOG_INTERVAL_MAX)
    return s


def grom_space(num_layers: int) -> dict[str, Param]:
    s = _sensor_base(num_layers)
    s["num_directions"] = _qln(U_MU, U_SIG)
    s["gate_hidden_dim"] = _qln(U_MU, U_SIG)
    s["intensity_hidden_dim"] = _qln(U_MU, U_SIG)
    s["optimization_steps"] = _qln(U_MU, U_SIG)
    s["learning_rate"] = _ln(U_MU, U_SIG)
    s["warmup_steps"] = _qln(U_MU, U_SIG)
    s["behavior_weight"] = _ln(U_MU, U_SIG)
    s["retain_weight"] = _uf(_C.SS_GROM_RETAIN_WEIGHT_LOW, _C.SS_GROM_RETAIN_WEIGHT_HIGH)
    s["sparse_weight"] = _ln(U_MU, U_SIG)
    s["smooth_weight"] = _ln(U_MU, U_SIG)
    s["independence_weight"] = _ln(U_MU, U_SIG)
    s["max_alpha"] = _ln(U_MU, U_SIG)
    s["gate_temperature"] = _ln(U_MU, U_SIG)
    s["max_grad_norm"] = _ln(U_MU, U_SIG)
    s["eta_min_factor"] = _uf(_C.SS_GROM_ETA_MIN_FACTOR_LOW, _C.SS_GROM_ETA_MIN_FACTOR_HIGH)
    s["linear_threshold"] = _uf(_C.SS_GROM_LINEAR_THRESH_LOW, _C.SS_GROM_LINEAR_THRESH_HIGH)
    s["adapt_cone_threshold"] = _uf(_C.SS_GROM_ADAPT_CONE_LOW, _C.SS_GROM_ADAPT_CONE_HIGH)
    s["adapt_manifold_threshold"] = _uf(_C.SS_GROM_ADAPT_MANIFOLD_LOW, _C.SS_GROM_ADAPT_MANIFOLD_HIGH)
    s["adapt_linear_directions"] = _ri(_C.SS_GROM_ADAPT_LINEAR_DIR_MIN, _C.SS_GROM_ADAPT_LINEAR_DIR_MAX)
    s["adapt_complex_directions"] = _ri(_C.SS_GROM_ADAPT_COMPLEX_DIR_MIN, _C.SS_GROM_ADAPT_COMPLEX_DIR_MAX)
    s["adapt_max_directions"] = _ri(_C.SS_GROM_ADAPT_MAX_DIR_MIN, _C.SS_GROM_ADAPT_MAX_DIR_MAX)
    s["significant_directions_default"] = _ri(_C.SS_GROM_SIG_DIR_DEFAULT_MIN, _C.SS_GROM_SIG_DIR_DEFAULT_MAX)
    s["min_adapted_directions"] = _ri(_C.SS_GROM_MIN_ADAPTED_DIR_MIN, _C.SS_GROM_MIN_ADAPTED_DIR_MAX)
    s["caa_similarity_skip"] = _uf(_C.SS_GROM_CAA_SIM_SKIP_LOW, _C.SS_GROM_CAA_SIM_SKIP_HIGH)
    s["contrastive_margin"] = _uf(_C.SS_GROM_CONTRASTIVE_MARGIN_LOW, _C.SS_GROM_CONTRASTIVE_MARGIN_HIGH)
    s["contrastive_weight"] = _ln(U_MU, U_SIG)
    s["utility_weight"] = _ln(U_MU, U_SIG)
    s["concentration_weight"] = _ln(U_MU, U_SIG)
    s["gate_warmup_weight"] = _ln(U_MU, U_SIG)
    s["caa_alignment_weight"] = _ln(U_MU, U_SIG)
    s["gate_dim_min"] = _ri(_C.SS_GROM_DIM_MIN, _C.SS_GROM_DIM_MAX)
    s["gate_dim_max"] = _ri(_C.SS_GROM_DIM_MIN, _C.SS_GROM_DIM_MAX)
    s["gate_dim_divisor"] = _ri(_C.SS_GROM_DIM_DIVISOR_MIN, _C.SS_GROM_DIM_DIVISOR_MAX)
    s["intensity_dim_min"] = _ri(_C.SS_GROM_DIM_MIN, _C.SS_GROM_DIM_MAX)
    s["intensity_dim_max"] = _ri(_C.SS_GROM_DIM_MIN, _C.SS_GROM_DIM_MAX)
    s["intensity_dim_divisor"] = _ri(_C.SS_GROM_DIM_DIVISOR_MIN, _C.SS_GROM_DIM_DIVISOR_MAX)
    s["gate_shrink_factor"] = _ri(_C.SS_GROM_SHRINK_MIN, _C.SS_GROM_SHRINK_MAX)
    s["weight_decay"] = _ln(U_MU, U_SIG)
    s["min_cosine_sim"] = _uf(_C.SS_GROM_MIN_COS_LOW, _C.SS_GROM_MIN_COS_HIGH)
    s["max_cosine_sim"] = _uf(_C.SS_GROM_MAX_COS_LOW, _C.SS_GROM_MAX_COS_HIGH)
    s["create_noise_scale"] = _ln(U_MU, U_SIG)
    s["create_gate_threshold"] = _uf(_C.SS_GROM_GATE_THRESH_LOW, _C.SS_GROM_GATE_THRESH_HIGH)
    s["log_interval"] = _ri(_C.SS_GROM_LOG_INTERVAL_MIN, _C.SS_GROM_LOG_INTERVAL_MAX)
    return s


def nurt_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["num_dims"] = _qln(U_MU, U_SIG)
    s["max_concept_dim"] = _qln(U_MU, U_SIG)
    s["variance_threshold"] = _uf(_C.SS_NURT_VAR_THRESH_LOW, _C.SS_NURT_VAR_THRESH_HIGH)
    s["training_epochs"] = _qln(U_MU, U_SIG)
    s["lr"] = _ln(U_MU, U_SIG)
    s["lr_min"] = _ln(U_MU, U_SIG)
    s["num_integration_steps"] = _ri(_C.SS_NURT_INTEGRATION_STEPS_MIN, _C.SS_NURT_INTEGRATION_STEPS_MAX)
    s["t_max"] = _uf(_C.SS_NURT_T_MAX_LOW, _C.SS_NURT_T_MAX_HIGH)
    s["flow_hidden_dim"] = _ri(_C.SS_NURT_FLOW_HIDDEN_DIM_MIN, _C.SS_NURT_FLOW_HIDDEN_DIM_MAX)
    s["weight_decay"] = _ln(U_MU, U_SIG)
    s["max_grad_norm"] = _ln(U_MU, U_SIG)
    return s


def szlak_space(num_layers: int) -> dict[str, Param]:
    s = _deferred_base(num_layers)
    s["sinkhorn_reg"] = _ln(U_MU, U_SIG)
    s["max_iter"] = _ri(_C.SS_SZLAK_MAX_ITER_MIN, _C.SS_SZLAK_MAX_ITER_MAX)
    s["inference_k"] = _ri(_C.SS_SZLAK_INFERENCE_K_MIN, _C.SS_SZLAK_INFERENCE_K_MAX)
    return s


def wicher_space(num_layers: int) -> dict[str, Param]:
    s = _base(num_layers)
    s["concept_dim"] = _Cat(choices=list(_C.SS_WICHER_CONCEPT_DIMS))
    s["variance_threshold"] = _uf(_C.SS_WICHER_VAR_THRESH_LOW, _C.SS_WICHER_VAR_THRESH_HIGH)
    s["num_steps"] = _ri(_C.SS_WICHER_NUM_STEPS_MIN, _C.SS_WICHER_NUM_STEPS_MAX)
    s["alpha"] = _ln(U_MU, U_SIG)
    s["eta"] = _uf(_C.SS_WICHER_ETA_LOW, _C.SS_WICHER_ETA_HIGH)
    s["beta"] = _uf(_C.SS_WICHER_BETA_LOW, _C.SS_WICHER_BETA_HIGH)
    s["alpha_decay"] = _uf(_C.SS_WICHER_ALPHA_DECAY_LOW, _C.SS_WICHER_ALPHA_DECAY_HIGH)
    s["solver"] = _Cat(choices=list(_C.WICHER_SOLVER_CHOICES))
    return s


def przelom_space(num_layers: int) -> dict[str, Param]:
    s = _deferred_base(num_layers)
    s["epsilon"] = _ln(U_MU, U_SIG)
    s["target_mode"] = _Cat(choices=list(_C.SS_PRZELOM_TARGET_MODES))
    s["regularization"] = _ln(U_MU, U_SIG)
    s["inference_k"] = _ri(_C.SS_PRZELOM_INFERENCE_K_MIN, _C.SS_PRZELOM_INFERENCE_K_MAX)
    return s


def get_method_space(method: str, num_layers: int) -> dict[str, Param]:
    """Look up the search space for a method by name."""
    dispatch = {
        "CAA": caa_space, "OSTRZE": ostrze_space, "MLP": mlp_space,
        "TECZA": tecza_space, "TETNO": tetno_space, "GROM": grom_space,
        "NURT": nurt_space, "SZLAK": szlak_space,
        "WICHER": wicher_space, "PRZELOM": przelom_space,
    }
    fn = dispatch.get(method.upper())
    if fn is None:
        raise ValueError(f"No search space for method: {method}")
    return fn(num_layers)
