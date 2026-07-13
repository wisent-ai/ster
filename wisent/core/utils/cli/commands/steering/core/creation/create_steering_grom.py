"""GROM steering object creation helper."""
from __future__ import annotations
from dataclasses import replace

from wisent.core.control.steering_methods.configs.optimal import get_optimal


def _require_arg(args, attr_name):
    val = getattr(args, attr_name, None)
    if val is None:
        raise ValueError(
            f"Parameter '{attr_name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return val




def _create_grom_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
    log_interval: int,
    *,
    gate_dim_min: int,
    gate_dim_max: int,
    gate_dim_divisor: int,
    gate_shrink_factor: int,
    intensity_dim_min: int,
    intensity_dim_max: int,
    intensity_dim_divisor: int,
    create_noise_scale: float,
    create_gate_threshold: float,
) -> GROMSteeringObject:
    """Train the canonical GROM method and adapt its result for persistence."""
    from wisent.core.control.steering_methods.methods.grom import GROMMethod
    from wisent.core.control.steering_methods._steering_object_grom import (
        GROMSteeringObject,
    )
    from wisent.core.utils.cli.commands.steering.core.creation.create_steering_helpers import (
        _pair_set_from_layer_activations,
    )

    if metadata.extraction_component != "residual_stream":
        raise ValueError(
            "GROM requires extraction_component='residual_stream'"
        )
    sensor_layer = int(_require_arg(args, "sensor_layer"))
    steering_value = _require_arg(args, "steering_layers")
    if isinstance(steering_value, str):
        steering_layers = [
            int(value.strip()) for value in steering_value.split(",") if value.strip()
        ]
    else:
        steering_layers = [int(value) for value in steering_value]
    steering_layers = sorted(set(steering_layers))
    if not steering_layers:
        raise ValueError("Parameter 'steering_layers' must contain at least one layer")
    if sensor_layer >= min(steering_layers):
        raise ValueError(
            "GROM sensor_layer must be strictly earlier than every steering layer"
        )

    required_layers = sorted({sensor_layer, *steering_layers})
    available = {int(layer) for layer in available_layers}
    missing = [layer for layer in required_layers if layer not in available]
    if missing:
        raise ValueError(f"Missing required GROM activation layers: {missing}")
    pair_set = _pair_set_from_layer_activations(
        layer_activations, required_layers, name="grom",
    )

    method = GROMMethod(
        sensor_layer=sensor_layer,
        steering_layers=steering_layers,
        num_directions=_require_arg(args, "grom_num_directions"),
        gate_hidden_dim=getattr(args, "grom_gate_hidden_dim", None),
        intensity_hidden_dim=getattr(args, "grom_intensity_hidden_dim", None),
        optimization_steps=_require_arg(args, "grom_optimization_steps"),
        learning_rate=_require_arg(args, "grom_learning_rate"),
        warmup_steps=_require_arg(args, "grom_warmup_steps"),
        behavior_weight=_require_arg(args, "grom_behavior_weight"),
        retain_weight=_require_arg(args, "grom_retain_weight"),
        sparse_weight=_require_arg(args, "grom_sparse_weight"),
        smooth_weight=_require_arg(args, "grom_smooth_weight"),
        independence_weight=_require_arg(args, "grom_independence_weight"),
        max_alpha=_require_arg(args, "grom_max_alpha"),
        gate_temperature=_require_arg(args, "grom_gate_temperature"),
        max_grad_norm=_require_arg(args, "grom_max_grad_norm"),
        eta_min_factor=_require_arg(args, "grom_eta_min_factor"),
        linear_threshold=_require_arg(args, "grom_linear_threshold"),
        adapt_cone_threshold=_require_arg(args, "grom_adapt_cone_threshold"),
        adapt_manifold_threshold=_require_arg(args, "grom_adapt_manifold_threshold"),
        adapt_linear_directions=_require_arg(args, "grom_adapt_linear_directions"),
        adapt_complex_directions=_require_arg(args, "grom_adapt_complex_directions"),
        adapt_max_directions=_require_arg(args, "grom_adapt_max_directions"),
        significant_directions_default=_require_arg(args, "grom_significant_directions_default"),
        min_adapted_directions=_require_arg(args, "grom_min_adapted_directions"),
        caa_similarity_skip=_require_arg(args, "grom_caa_similarity_skip"),
        contrastive_margin=_require_arg(args, "grom_contrastive_margin"),
        contrastive_weight=_require_arg(args, "grom_contrastive_weight"),
        utility_weight=_require_arg(args, "grom_utility_weight"),
        concentration_weight=_require_arg(args, "grom_concentration_weight"),
        gate_warmup_weight=_require_arg(args, "grom_gate_warmup_weight"),
        caa_alignment_weight=_require_arg(args, "grom_caa_alignment_weight"),
        weight_decay=_require_arg(args, "grom_weight_decay"),
        min_cosine_similarity=_require_arg(args, "grom_min_cosine_sim"),
        max_cosine_similarity=_require_arg(args, "grom_max_cosine_sim"),
        gate_dim_min=gate_dim_min,
        gate_dim_max=gate_dim_max,
        gate_dim_divisor=gate_dim_divisor,
        intensity_dim_min=intensity_dim_min,
        intensity_dim_max=intensity_dim_max,
        intensity_dim_divisor=intensity_dim_divisor,
        gate_shrink_factor=gate_shrink_factor,
        create_noise_scale=create_noise_scale,
        create_gate_threshold=create_gate_threshold,
        log_interval=log_interval,
        normalize=getattr(args, "normalize", get_optimal("normalize")),
    )
    result = method.train_grom(pair_set)

    directions = {
        int(str(layer).split("_")[-1]): value
        for layer, value in result.directions.items()
    }
    direction_weights = {
        int(str(layer).split("_")[-1]): value
        for layer, value in result.direction_weights.items()
    }
    layer_order = [int(str(layer).split("_")[-1]) for layer in result.layer_order]
    result_sensor_layer = int(str(result.sensor_layer).split("_")[-1])
    if result_sensor_layer != sensor_layer:
        raise ValueError(
            "GROM method result sensor_layer does not match configured sensor_layer"
        )
    expected_layers = set(steering_layers)
    if (
        set(directions) != expected_layers
        or set(direction_weights) != expected_layers
        or layer_order != steering_layers
    ):
        raise ValueError(
            "GROM method result layers do not exactly match configured steering_layers"
        )
    return GROMSteeringObject(
        metadata=replace(
            metadata, layers=steering_layers, sensor_layer=sensor_layer,
        ),
        directions=directions,
        direction_weights=direction_weights,
        gate_network=result.gate_network,
        intensity_network=result.intensity_network,
        layer_order=layer_order,
        sensor_layer=sensor_layer,
        gate_temperature=result.gate_temperature,
        max_alpha=method.config.max_alpha,
    )
