"""
Main parser setup for Wisent CLI.

This module imports and combines all command-specific parsers into a single
argparse parser for the Wisent CLI.
"""

import argparse

# Import from reorganized subdirectories
from wisent.core.utils.config_tools.parser_arguments.other.utilities import setup_tasks_parser
from wisent.core.utils.config_tools.parser_arguments.generation.pairs import (
    setup_generate_pairs_parser,
    setup_generate_pairs_from_task_parser,
    setup_generate_responses_parser,
)
from wisent.core.utils.config_tools.parser_arguments.analysis.diagnostics import (
    setup_diagnose_pairs_parser,
    setup_diagnose_vectors_parser,
)
from wisent.core.utils.config_tools.parser_arguments.training import (
    setup_get_activations_parser,
    setup_synthetic_parser,
    setup_train_unified_goodness_parser,
)
from wisent.core.utils.config_tools.parser_arguments.configuration import (
    setup_create_steering_object_parser,
    setup_model_config_parser,
    setup_configure_model_parser,
    setup_inference_config_parser,
)
from wisent.core.utils.config_tools.parser_arguments.generation.vectors import (
    setup_generate_vector_from_task_parser,
    setup_generate_vector_from_synthetic_parser,
    setup_generate_vector_parser,
)
from wisent.core.utils.config_tools.parser_arguments.other.utilities import (
    setup_test_nonsense_parser,
    setup_monitor_parser,
)
from wisent.core.utils.config_tools.parser_arguments.other import (
    setup_agent_parser,
    setup_modify_weights_parser,
)
from wisent.core.utils.config_tools.parser_arguments.optimization.steering import (
    setup_classification_optimizer_parser,
    setup_steering_optimizer_parser,
    setup_sample_size_optimizer_parser,
)
from wisent.core.utils.config_tools.parser_arguments.optimization import (
    setup_optimize_all_parser,
    setup_optimize_parser,
)
from wisent.core.utils.config_tools.parser_arguments.optimization.weights import (
    setup_optimization_cache_parser,
    setup_optimize_weights_parser,
)
from wisent.core.utils.config_tools.parser_arguments.other.steering import (
    setup_multi_steer_parser, setup_steering_viz_parser,
    setup_discover_steering_parser, setup_find_best_method_parser,
)
from wisent.core.utils.config_tools.parser_arguments.evaluation import (
    setup_evaluate_parser, setup_evaluate_responses_parser,
    setup_evaluate_refusal_parser, setup_verify_steering_parser,
)
from wisent.core.utils.config_tools.parser_arguments.concepts import setup_axbench_parser
from wisent.core.utils.config_tools.parser_arguments.analysis import (
    setup_check_linearity_parser, setup_cluster_benchmarks_parser,
    setup_geometry_search_parser, setup_zwiad_parser, setup_compare_steering_parser,
)
from wisent.core.utils.config_tools.parser_arguments.data import setup_migrate_activations_parser

def setup_parser() -> argparse.ArgumentParser:
    """Set up the main CLI parser with subcommands."""
    parser = argparse.ArgumentParser(description="Wisent: Advanced AI Safety and Alignment Toolkit")

    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tasks command (main evaluation pipeline)
    tasks_parser = subparsers.add_parser("tasks", help="Run evaluation tasks")
    setup_tasks_parser(tasks_parser)

    # Generate pairs command
    generate_parser = subparsers.add_parser("generate-pairs", help="Generate synthetic contrastive pairs")
    setup_generate_pairs_parser(generate_parser)

    # Diagnose pairs command
    diagnose_parser = subparsers.add_parser("diagnose-pairs", help="Diagnose and analyze existing contrastive pairs")
    setup_diagnose_pairs_parser(diagnose_parser)

    # Generate pairs from task command
    generate_from_task_parser = subparsers.add_parser("generate-pairs-from-task", help="Generate contrastive pairs from lm-eval task")

    # Get activations command
    get_activations_parser = subparsers.add_parser("get-activations", help="Collect activations from contrastive pairs")
    setup_get_activations_parser(get_activations_parser)
    setup_generate_pairs_from_task_parser(generate_from_task_parser)

    # Diagnose vectors command
    diagnose_vectors_parser = subparsers.add_parser("diagnose-vectors", help="Diagnose and analyze existing steering vectors")
    setup_diagnose_vectors_parser(diagnose_vectors_parser)

    # Create steering vector command (creates full steering objects with method-specific components)
    create_steering_parser = subparsers.add_parser("create-steering-vector", help="Create steering vectors from enriched pairs")
    setup_create_steering_object_parser(create_steering_parser)

    # Generate vector from task command (full pipeline)
    generate_vector_from_task_parser = subparsers.add_parser("generate-vector-from-task", help="Generate steering vector from task (full pipeline)")
    setup_generate_vector_from_task_parser(generate_vector_from_task_parser)

    # Generate vector from synthetic command (full pipeline)
    generate_vector_from_synthetic_parser = subparsers.add_parser("generate-vector-from-synthetic", help="Generate steering vector from synthetic pairs (full pipeline)")
    setup_generate_vector_from_synthetic_parser(generate_vector_from_synthetic_parser)

    # Synthetic command (generate + train + test)
    synthetic_parser = subparsers.add_parser("synthetic", help="Run synthetic contrastive pair pipeline")
    setup_synthetic_parser(synthetic_parser)

    # Test nonsense detection command
    test_nonsense_parser = subparsers.add_parser("test-nonsense", help="Test nonsense detection system")
    setup_test_nonsense_parser(test_nonsense_parser)

    # Monitor command for performance monitoring
    monitor_parser = subparsers.add_parser("monitor", help="Performance monitoring and system information")
    setup_monitor_parser(monitor_parser)

    # Agent command for autonomous agent interaction
    agent_parser = subparsers.add_parser("agent", help="Interact with autonomous agent")
    setup_agent_parser(agent_parser)

    # Model configuration command for managing optimal parameters
    model_config_parser = subparsers.add_parser("model-config", help="Manage model-specific optimal parameters")
    setup_model_config_parser(model_config_parser)

    # Configure model command for setting up new/unsupported models
    configure_model_parser = subparsers.add_parser(
        "configure-model", help="Configure tokens and layer access for unsupported models"
    )
    setup_configure_model_parser(configure_model_parser)

    # Classification optimization command for finding optimal classification parameters
    classification_optimizer_parser = subparsers.add_parser(
        "optimize-classification", help="Optimize classification parameters across all tasks"
    )
    setup_classification_optimizer_parser(classification_optimizer_parser)

    # Steering optimization command for finding optimal steering parameters
    steering_optimizer_parser = subparsers.add_parser(
        "optimize-steering", help="Optimize steering parameters for different methods"
    )
    setup_steering_optimizer_parser(steering_optimizer_parser)

    # Sample size optimization command for finding optimal training sample sizes
    sample_size_optimizer_parser = subparsers.add_parser(
        "optimize-sample-size", help="Find optimal training sample size for classifiers"
    )
    setup_sample_size_optimizer_parser(sample_size_optimizer_parser)

    # Full optimization command that runs all optimizations for a model
    optimize_all_parser = subparsers.add_parser(
        "optimize-all", help="Run all optimizations: classification, steering, and weight modification"
    )
    setup_optimize_all_parser(optimize_all_parser)
    
    # Alias: 'optimize' is the same as 'optimize-all'
    optimize_parser = subparsers.add_parser(
        "optimize", help="Run all optimizations (alias for optimize-all)"
    )
    setup_optimize_all_parser(optimize_parser)

    # Generate vector command for creating steering vectors without tasks
    generate_vector_parser = subparsers.add_parser(
        "generate-vector", help="Generate steering vectors from contrastive pairs (file or description)"
    )
    setup_generate_vector_parser(generate_vector_parser)

    # Multi-vector steering command for combining multiple vectors at inference time
    multi_steer_parser = subparsers.add_parser(
        "multi-steer", help="Combine multiple steering vectors dynamically at inference time"
    )
    setup_multi_steer_parser(multi_steer_parser)

    # Single-prompt evaluation command for real-time steering assessment
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate single prompt with steering vector and return quality scores"
    )
    setup_evaluate_parser(evaluate_parser)

    # Generate responses command for generating model responses to task questions
    generate_responses_parser = subparsers.add_parser(
        "generate-responses", help="Generate model responses to questions from a task"
    )
    setup_generate_responses_parser(generate_responses_parser)

    # Evaluate responses command for evaluating generated responses
    evaluate_responses_parser = subparsers.add_parser(
        "evaluate-responses", help="Evaluate generated responses using embedded evaluator"
    )
    setup_evaluate_responses_parser(evaluate_responses_parser)

    # Modify weights command for permanent weight modification
    modify_weights_parser = subparsers.add_parser(
        "modify-weights", help="Permanently modify model weights using steering vectors (directional projection or additive)"
    )
    setup_modify_weights_parser(modify_weights_parser)

    # Evaluate refusal command for measuring model refusal rate
    evaluate_refusal_parser = subparsers.add_parser(
        "evaluate-refusal", help="Evaluate model refusal rate on potentially harmful prompts"
    )
    setup_evaluate_refusal_parser(evaluate_refusal_parser)

    # Inference config command for viewing/updating generation settings
    inference_config_parser = subparsers.add_parser(
        "inference-config", help="View and update inference generation settings"
    )
    setup_inference_config_parser(inference_config_parser)

    # Optimization cache command for managing cached optimization results
    optimization_cache_parser = subparsers.add_parser(
        "optimization-cache", help="Manage cached optimization results (list, show, delete, clear, export, import)"
    )
    setup_optimization_cache_parser(optimization_cache_parser)

    # Optimize weights command for finding optimal weight modification parameters
    optimize_weights_parser = subparsers.add_parser(
        "optimize-weights", help="Optimize weight modification parameters for any trait/task using Optuna"
    )
    setup_optimize_weights_parser(optimize_weights_parser)

    # Unified goodness training command - train single vector from ALL benchmarks
    unified_goodness_parser = subparsers.add_parser(
        "train-unified-goodness",
        help="Train a single 'goodness' steering vector from pooled multi-benchmark data"
    )
    setup_train_unified_goodness_parser(unified_goodness_parser)

    # Check linearity command - check if a representation is linear
    check_linearity_parser = subparsers.add_parser(
        "check-linearity",
        help="Check if a representation is linear (can be captured by single direction)"
    )
    setup_check_linearity_parser(check_linearity_parser)

    # Cluster benchmarks command - cluster benchmarks by direction similarity
    cluster_benchmarks_parser = subparsers.add_parser(
        "cluster-benchmarks",
        help="Cluster benchmarks by direction similarity with geometry analysis"
    )
    setup_cluster_benchmarks_parser(cluster_benchmarks_parser)

    # Geometry search command - search for unified goodness direction across all benchmarks
    geometry_search_parser = subparsers.add_parser(
        "geometry-search",
        help="Search for unified goodness direction across benchmarks (analyzes structure: linear/cone/orthogonal)"
    )
    setup_geometry_search_parser(geometry_search_parser)

    # Verify steering command - verify activation alignment at inference time
    verify_steering_parser = subparsers.add_parser(
        "verify-steering",
        help="Verify that steered model activations are correctly aligned with steering direction"
    )
    setup_verify_steering_parser(verify_steering_parser)

    # Zwiad command - geometry analysis with concept decomposition on database activations
    zwiad_parser = subparsers.add_parser(
        "zwiad",
        help="Run Zwiad geometry analysis with concept decomposition on database activations"
    )
    setup_zwiad_parser(zwiad_parser)

    # Steering visualization command - show steering effect on activation space
    steering_viz_parser = subparsers.add_parser(
        "steering-viz",
        help="Visualize steering effect on activation space (base vs steered relative to pos/neg)"
    )
    setup_steering_viz_parser(steering_viz_parser)

    discover_steering_parser = subparsers.add_parser("discover-steering", help="Discover optimal steering directions")
    setup_discover_steering_parser(discover_steering_parser)
    migrate_activations_parser = subparsers.add_parser("migrate-activations", help="Migrate activation data to HuggingFace Hub")
    setup_migrate_activations_parser(migrate_activations_parser)
    compare_steering_parser = subparsers.add_parser("compare-steering", help="Compare steering objects across traits")
    setup_compare_steering_parser(compare_steering_parser)
    find_best_parser = subparsers.add_parser("find-best-method", help="Find the best steering method for a benchmark")
    setup_find_best_method_parser(find_best_parser)
    axbench_parser = subparsers.add_parser("axbench", help="Run the AxBench concept steering/detection benchmark")
    setup_axbench_parser(axbench_parser)
    return parser
