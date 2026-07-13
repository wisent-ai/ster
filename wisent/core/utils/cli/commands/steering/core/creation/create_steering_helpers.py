"""TECZA and TETNO steering object creation helpers."""
from __future__ import annotations
from dataclasses import replace

import torch

from wisent.core.control.steering_methods.configs.optimal import get_optimal


def _require_arg(args, attr_name):
    """Get required arg or raise clear error."""
    val = getattr(args, attr_name, None)
    if val is None:
        raise ValueError(
            f"Parameter '{attr_name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return val


def _create_tecza_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> TECZASteeringObject:
    """Create TECZA steering object with multiple directions."""
    from wisent.core.control.steering_methods.methods.advanced import TECZAMethod

    _r = _require_arg
    method = TECZAMethod(
        num_directions=_r(args, 'tecza_num_directions'),
        optimization_steps=_r(args, 'tecza_optimization_steps'),
        learning_rate=_r(args, 'tecza_learning_rate'),
        retain_weight=_r(args, 'tecza_retain_weight'),
        independence_weight=_r(args, 'tecza_independence_weight'),
        min_cosine_similarity=_r(args, 'tecza_min_cosine_sim'),
        max_cosine_similarity=_r(args, 'tecza_max_cosine_sim'),
        variance_threshold=_r(args, 'tecza_variance_threshold'),
        marginal_threshold=_r(args, 'tecza_marginal_threshold'),
        max_directions=_r(args, 'tecza_max_directions'),
        ablation_weight=_r(args, 'tecza_ablation_weight'),
        addition_weight=_r(args, 'tecza_addition_weight'),
        separation_margin=_r(args, 'tecza_separation_margin'),
        perturbation_scale=_r(args, 'tecza_perturbation_scale'),
        universal_basis_noise=_r(args, 'tecza_universal_basis_noise'),
        log_interval=_r(args, 'tecza_log_interval'),
        normalize=getattr(args, 'normalize', get_optimal("normalize")),
    )
    
    directions = {}
    direction_weights = {}
    
    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        
        if not pos_list or not neg_list:
            continue
        
        # Stack activations
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        # Train directions
        layer_dirs, meta = method._train_layer_directions(pos_tensor, neg_tensor, layer_str, log_interval=method.log_interval)
        
        layer_int = int(layer_str)
        directions[layer_int] = layer_dirs
        # Equal weights by default
        direction_weights[layer_int] = torch.ones(layer_dirs.shape[0]) / layer_dirs.shape[0]
        
        print(f"   Layer {layer_str}: {layer_dirs.shape[0]} directions, avg_cosine={meta['avg_cosine_similarity']:.3f}")
    
    from wisent.core.control.steering_methods._steering_object_advanced import TECZASteeringObject
    return TECZASteeringObject(
        metadata=metadata,
        directions=directions,
        direction_weights=direction_weights,
        primary_only=False,
    )


def _pair_set_from_layer_activations(
    layer_activations: dict,
    required_layers: list[int],
    *,
    name: str,
):
    """Build a method input from one complete, identity-aligned pair set."""
    from wisent.core.primitives.contrastive_pairs.core.io.response import (
        NegativeResponse,
        PositiveResponse,
    )
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.primitives.model_interface.core.activations.core.atoms import (
        LayerActivations,
    )

    layer_keys = [str(layer) for layer in required_layers]
    indexed: dict[tuple[str, str], dict[str | int, torch.Tensor]] = {}
    shared_pair_ids: list[str | int] | None = None
    shared_pair_id_set: set[str | int] | None = None
    for layer_key in layer_keys:
        if layer_key not in layer_activations:
            raise ValueError(f"Missing required activation layer {layer_key}")
        layer_data = layer_activations[layer_key]
        for side in ("positive", "negative"):
            activations = layer_data.get(side, [])
            pair_ids = layer_data.get(f"{side}_pair_ids")
            if not activations:
                raise ValueError(
                    f"Empty {side} activations at required layer {layer_key}"
                )
            if pair_ids is None:
                raise ValueError(
                    f"Required activation layer {layer_key} is missing "
                    f"{side}_pair_ids; pair identity is required for {name.upper()}"
                )
            if len(pair_ids) != len(activations):
                raise ValueError(
                    f"Required activation layer {layer_key} has {len(activations)} "
                    f"{side} rows but {len(pair_ids)} pair identities"
                )
            invalid = [
                pair_id for pair_id in pair_ids
                if isinstance(pair_id, bool)
                or not isinstance(pair_id, (str, int))
                or (isinstance(pair_id, str) and not pair_id.strip())
            ]
            if invalid:
                raise ValueError(
                    f"Required activation layer {layer_key} has invalid {side} "
                    f"pair identities: {invalid!r}"
                )
            pair_id_set = set(pair_ids)
            if len(pair_id_set) != len(pair_ids):
                raise ValueError(
                    f"Required activation layer {layer_key} has duplicate {side} "
                    "pair identities"
                )
            if shared_pair_ids is None:
                shared_pair_ids = list(pair_ids)
                shared_pair_id_set = pair_id_set
            elif pair_id_set != shared_pair_id_set:
                missing = sorted(shared_pair_id_set - pair_id_set, key=repr)
                extra = sorted(pair_id_set - shared_pair_id_set, key=repr)
                raise ValueError(
                    "Required activation layers do not contain one complete shared "
                    f"pair identity set: layer {layer_key} {side} is missing {missing!r} "
                    f"and has extra {extra!r}"
                )
            indexed[(layer_key, side)] = dict(zip(pair_ids, activations))

    if shared_pair_ids is None:
        raise ValueError(f"No required activation layers supplied for {name.upper()}")
    pair_set = ContrastivePairSet(name=name)
    for pair_id in shared_pair_ids:
        positive = LayerActivations({
            key: indexed[(key, "positive")][pair_id] for key in layer_keys
        })
        negative = LayerActivations({
            key: indexed[(key, "negative")][pair_id] for key in layer_keys
        })
        pair_set.add(ContrastivePair(
            prompt=f"{name} activation pair {pair_id!r}",
            positive_response=PositiveResponse(
                model_response="positive activation", layers_activations=positive,
            ),
            negative_response=NegativeResponse(
                model_response="negative activation", layers_activations=negative,
            ),
            metadata={"pair_id": pair_id},
        ))
    return pair_set


def _create_tetno_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> TETNOSteeringObject:
    """Train the canonical TETNO method and adapt its result for persistence."""
    from wisent.core.control.steering_methods.methods.advanced import TETNOMethod
    from wisent.core.control.steering_methods._steering_object_advanced import (
        TETNOSteeringObject,
    )

    if metadata.extraction_component != "residual_stream":
        raise ValueError(
            "TETNO requires extraction_component='residual_stream'"
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
            "TETNO sensor_layer must be strictly earlier than every steering layer"
        )

    required_layers = sorted({sensor_layer, *steering_layers})
    available = {int(layer) for layer in available_layers}
    missing = [layer for layer in required_layers if layer not in available]
    if missing:
        raise ValueError(f"Missing required TETNO activation layers: {missing}")
    pair_set = _pair_set_from_layer_activations(
        layer_activations, required_layers, name="tetno",
    )

    method = TETNOMethod(
        sensor_layer=sensor_layer,
        steering_layers=steering_layers,
        per_layer_scaling=getattr(
            args, "tetno_per_layer_scaling", get_optimal("per_layer_scaling")
        ),
        condition_threshold=_require_arg(args, "tetno_condition_threshold"),
        gate_temperature=_require_arg(args, "tetno_gate_temperature"),
        learn_threshold=getattr(
            args, "tetno_learn_threshold", get_optimal("learn_threshold")
        ),
        use_entropy_scaling=getattr(
            args, "tetno_use_entropy_scaling", get_optimal("use_entropy_scaling")
        ),
        entropy_floor=_require_arg(args, "tetno_entropy_floor"),
        entropy_ceiling=_require_arg(args, "tetno_entropy_ceiling"),
        max_alpha=_require_arg(args, "tetno_max_alpha"),
        optimization_steps=_require_arg(args, "tetno_optimization_steps"),
        learning_rate=_require_arg(args, "tetno_learning_rate"),
        threshold_search_steps=_require_arg(args, "tetno_threshold_search_steps"),
        condition_margin=_require_arg(args, "tetno_condition_margin"),
        min_layer_scale=_require_arg(args, "tetno_min_layer_scale"),
        log_interval=_require_arg(args, "tetno_log_interval"),
        normalize=getattr(args, "normalize", get_optimal("normalize")),
    )
    result = method.train_tetno(pair_set)

    behavior_vectors = {
        int(str(layer).split("_")[-1]): vector
        for layer, vector in result.behavior_vectors.items()
    }
    layer_scales = {
        int(str(layer).split("_")[-1]): scale
        for layer, scale in result.layer_scales.items()
    }
    expected_layers = set(steering_layers)
    if set(behavior_vectors) != expected_layers or set(layer_scales) != expected_layers:
        raise ValueError(
            "TETNO method result layers do not exactly match configured steering_layers"
        )
    return TETNOSteeringObject(
        metadata=replace(
            metadata, layers=steering_layers, sensor_layer=sensor_layer,
        ),
        behavior_vectors=behavior_vectors,
        condition_vector=result.condition_vector,
        sensor_layer=sensor_layer,
        threshold=result.optimal_threshold,
        layer_scales=layer_scales,
        gate_temperature=method.config.gate_temperature,
    )


