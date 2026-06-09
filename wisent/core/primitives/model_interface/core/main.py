"""
Main entry point for the Wisent CLI.

This module connects the argparse parser (wisent/core/parser_arguments/) to execution logic
and provides the main() function that serves as the CLI entry point.
"""

# Note: do NOT set os.environ["NUMBA_NUM_THREADS"] here. wisent/__init__.py
# documents why: numba's reload_config raises RuntimeError when the env value
# diverges from the already-launched thread count (this killed GCP runs with
# "Cannot set NUMBA_NUM_THREADS ... currently have 1, trying to set 4").
# Operators who want to constrain numba must export NUMBA_NUM_THREADS before
# launching python.

# Import pynndescent early before sklearn uses numba
import pynndescent  # noqa: F401

import sys
from pathlib import Path
from wisent.core.utils.config_tools.parser_arguments import setup_parser
from wisent.core.utils import print_banner
from wisent.core.utils.config_tools.constants import BANNER_WIDTH
import wisent.core.utils.cli as _cli


def _should_show_banner() -> bool:
    """Check if this is the first use and banner should be shown."""
    wisent_dir = Path.home() / ".wisent"
    banner_flag = wisent_dir / ".banner_shown"

    if banner_flag.exists():
        return False

    # First use - create the flag file
    wisent_dir.mkdir(parents=True, exist_ok=True)
    banner_flag.touch()
    return True


def main():
    """Main entry point for the Wisent CLI."""
    # Show banner only on first use
    if _should_show_banner():
        print_banner("Wisent CLI", width=BANNER_WIDTH, use_color=True)

    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, 'command') or args.command is None:
        parser.print_help()
        sys.exit(0)

    # Command-to-function dispatch map (modules lazy-loaded via cli.__getattr__)
    _COMMAND_MAP = {
        'tasks': 'execute_tasks',
        'generate-pairs': 'execute_generate_pairs',
        'diagnose-pairs': 'execute_diagnose_pairs',
        'generate-pairs-from-task': 'execute_generate_pairs_from_task',
        'get-activations': 'execute_get_activations',
        'diagnose-vectors': 'execute_diagnose_vectors',
        'create-steering-vector': 'execute_create_steering_object',
        'generate-vector-from-task': 'execute_generate_vector_from_task',
        'generate-vector-from-synthetic': 'execute_generate_vector_from_synthetic',
        'synthetic': 'execute_generate_vector_from_synthetic',
        'optimize-classification': 'execute_optimize_classification',
        'optimize-steering': 'execute_optimize_steering',
        'optimize-sample-size': 'execute_optimize_sample_size',
        'generate-responses': 'execute_generate_responses',
        'evaluate-responses': 'execute_evaluate_responses',
        'multi-steer': 'execute_multi_steer',
        'agent': 'execute_agent',
        'modify-weights': 'execute_modify_weights',
        'evaluate-refusal': 'execute_evaluate_refusal',
        'inference-config': 'execute_inference_config',
        'optimization-cache': 'execute_optimization_cache',
        'optimize-weights': 'execute_optimize_weights',
        'optimize-all': 'execute_optimize',
        'optimize': 'execute_optimize',
        'train-unified-goodness': 'execute_train_unified_goodness',
        'check-linearity': 'execute_check_linearity',
        'cluster-benchmarks': 'execute_cluster_benchmarks',
        'geometry-search': 'execute_geometry_search',
        'verify-steering': 'execute_verify_steering',
        'zwiad': 'execute_zwiad',
        'discover-steering': 'execute_discover_steering',
        'migrate-activations': 'execute_migrate_activations',
        'compare-steering': 'execute_compare_steering',
        'find-best-method': 'execute_find_best_method',
        'axbench': 'execute_axbench',
    }

    # Handle steering-viz specially (has per_concept variant)
    if args.command == 'steering-viz':
        if getattr(args, 'per_concept', False):
            func_name = 'execute_per_concept_steering_viz'
        else:
            func_name = 'execute_steering_viz'
    else:
        func_name = _COMMAND_MAP.get(args.command)

    if func_name is None:
        parser.error(f"Command '{args.command}' is not yet implemented")

    handler = getattr(_cli, func_name)
    handler(args)


if __name__ == '__main__':
    main()
