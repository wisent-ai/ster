"""Multi-direction method training for optimize-weights."""
import os

import torch


def _train_multi_direction_method(
    args,
    caa_vectors: dict[int, torch.Tensor],
    intermediate_dir: str,
    method: str,
) -> dict[int, torch.Tensor]:
    """Train GROM/TECZA/TETNO on pairs with activations and return combined directions.
    
    Args:
        args: Command line arguments
        caa_vectors: CAA vectors (used as fallback if training fails)
        intermediate_dir: Directory containing pairs with activations
        method: 'grom', 'tecza', or 'tetno'
        
    Returns:
        Combined steering vectors per layer
    """
    import glob
    
    # Find enriched pairs file (with activations)
    enriched_files = glob.glob(os.path.join(intermediate_dir, "*_with_activations.json"))
    
    if not enriched_files:
        print(f"   Warning: No enriched pairs found for {method}, using CAA vectors")
        return caa_vectors
    
    enriched_file = enriched_files[0]
    print(f"\n   Training {method.upper()} on enriched pairs...")
    print(f"   Pairs file: {enriched_file}")
    
    try:
        from wisent.core.primitives.contrastive_pairs.core.io.serialization import load_contrastive_pair_set
        from wisent.core.weight_modification.multi_direction import (
            MultiDirectionConfig,
            combine_directions,
        )
        
        # Load pair set with activations
        pair_set = load_contrastive_pair_set(enriched_file, return_backend="torch")
        print(f"   Loaded {len(pair_set.pairs)} pairs with activations")
        
        # Get config from args
        num_directions = args.num_directions
        combination_strategy = getattr(args, 'combination_strategy', None)
        if combination_strategy is None:
            raise ValueError("Parameter 'combination_strategy' is required.")
        optimization_steps = args.multi_optimization_steps
        if optimization_steps is None:
            raise ValueError("--multi-optimization-steps is required for multi-direction training")
        # Train the method
        if method == 'grom':
            from wisent.core.control.steering_methods.methods.grom import GROMMethod, GROMConfig
            config = GROMConfig(
                num_directions=num_directions,
                optimization_steps=optimization_steps,
                learning_rate=args.grom_learning_rate,
                warmup_steps=args.grom_warmup_steps,
                behavior_weight=args.grom_behavior_weight,
                retain_weight=args.grom_retain_weight,
                sparse_weight=args.grom_sparse_weight,
                smooth_weight=args.grom_smooth_weight,
                independence_weight=args.grom_independence_weight,
                max_alpha=args.grom_max_alpha,
                gate_temperature=args.grom_gate_temperature,
                min_cosine_similarity=args.grom_min_cosine_sim,
                max_cosine_similarity=args.grom_max_cosine_sim,
                weight_decay=args.grom_weight_decay,
                max_grad_norm=args.grom_max_grad_norm,
                eta_min_factor=args.grom_eta_min_factor,
                linear_threshold=args.grom_linear_threshold,
                adapt_cone_threshold=args.grom_adapt_cone_threshold,
                adapt_manifold_threshold=args.grom_adapt_manifold_threshold,
                adapt_linear_directions=args.grom_adapt_linear_directions,
                adapt_complex_directions=args.grom_adapt_complex_directions,
                adapt_max_directions=args.grom_adapt_max_directions,
                significant_directions_default=args.grom_significant_directions_default,
                min_adapted_directions=args.grom_min_adapted_directions,
                caa_similarity_skip=args.grom_caa_similarity_skip,
                contrastive_margin=args.grom_contrastive_margin,
                contrastive_weight=args.grom_contrastive_weight,
                utility_weight=args.grom_utility_weight,
                concentration_weight=args.grom_concentration_weight,
                gate_warmup_weight=args.grom_gate_warmup_weight,
                caa_alignment_weight=args.grom_caa_alignment_weight,
                gate_dim_min=args.grom_gate_dim_min,
                gate_dim_max=args.grom_gate_dim_max,
                gate_dim_divisor=args.grom_gate_dim_divisor,
                intensity_dim_min=args.grom_intensity_dim_min,
                intensity_dim_max=args.grom_intensity_dim_max,
                intensity_dim_divisor=args.grom_intensity_dim_divisor,
                gate_shrink_factor=args.grom_gate_shrink_factor,
                create_noise_scale=args.grom_create_noise_scale,
                create_gate_threshold=args.grom_create_gate_threshold,
            )
            trainer = GROMMethod(config=config, log_interval=args.grom_log_interval)
            result = trainer.train_grom(pair_set)
            directions = result.directions
            weights = result.direction_weights
            
        elif method == 'tecza':
            from wisent.core.control.steering_methods.methods.advanced import TECZAMethod
            trainer = TECZAMethod(
                num_directions=num_directions, optimization_steps=optimization_steps,
                learning_rate=args.tecza_learning_rate, retain_weight=args.tecza_retain_weight,
                independence_weight=args.tecza_independence_weight, ablation_weight=args.tecza_ablation_weight,
                addition_weight=args.tecza_addition_weight, separation_margin=args.tecza_separation_margin,
                perturbation_scale=args.tecza_perturbation_scale, universal_basis_noise=args.tecza_universal_basis_noise,
                min_cosine_similarity=args.tecza_min_cosine_sim, max_cosine_similarity=args.tecza_max_cosine_sim,
                variance_threshold=args.tecza_variance_threshold, marginal_threshold=args.tecza_marginal_threshold,
                max_directions=args.tecza_max_directions, log_interval=args.tecza_log_interval,
            )
            result = trainer.train_tecza(pair_set)
            directions = result.directions
            weights = None  # TECZA doesn't have learned weights
            
        elif method == 'tetno':
            from wisent.core.control.steering_methods.methods.advanced import TETNOMethod, TETNOConfig
            config = TETNOConfig(
                optimization_steps=optimization_steps,
            )
            trainer = TETNOMethod(config=config)
            result = trainer.train_tetno(pair_set)
            # TETNO has single direction per layer
            directions = {k: v.unsqueeze(0) for k, v in result.behavior_vectors.items()}
            weights = {k: torch.tensor([result.layer_scales[k]])
                      for k in directions} if result.layer_scales else None
        
        print(f"   Trained {len(directions)} layers with {method.upper()}")
        
        # Combine directions into single vector per layer
        combined_vectors = {}
        for layer_name, layer_dirs in directions.items():
            # Get layer index
            try:
                layer_idx = int(layer_name.replace("layer_", "")) - 1
            except ValueError:
                layer_idx = int(layer_name) - 1
            
            layer_weights = weights.get(layer_name) if weights else None
            combined = combine_directions(layer_dirs, layer_weights, strategy=combination_strategy)
            combined_vectors[layer_idx] = combined
            
        print(f"   Combined into {len(combined_vectors)} steering vectors")
        print(f"   Combination strategy: {combination_strategy}")
        
        return combined_vectors
        
    except Exception as e:
        print(f"   Warning: {method.upper()} training failed: {e}")
        print(f"   Falling back to CAA vectors")
        return caa_vectors

