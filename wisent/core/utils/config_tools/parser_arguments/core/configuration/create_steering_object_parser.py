"""Parser for the create-steering-object command."""

import argparse
from wisent.core.control.steering_methods import SteeringMethodRegistry


def setup_create_steering_object_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the create-steering-object command parser.

    This command creates full steering objects that preserve method-specific
    components like gates, intensity networks, multi-directions, etc.
    """
    # Input/Output
    parser.add_argument(
        "enriched_pairs_file",
        type=str,
        help="Path to JSON file containing contrastive pairs with activations"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for steering object (.pt or .json)"
    )

    # Steering method
    parser.add_argument(
        "--method",
        type=str,
        choices=["caa", "ostrze", "mlp", "tecza", "tetno", "grom", "nurt"],
        required=True,
        help="Steering method to use"
    )

    # Layer selection
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer(s) to create steering vectors for. Can be single (e.g., '16'), "
             "comma-separated (e.g., '12,14,16'), or range (e.g., '12-18'). "
             "If not specified, creates vectors for all layers."
    )

    # Common parameters
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize steering vectors (default: True)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Do not L2-normalize steering vectors"
    )

    # MLP parameters
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for MLP classifier (required at runtime)"
    )
    parser.add_argument(
        "--mlp-num-layers",
        type=int,
        default=None,
        help="Number of hidden layers in MLP (required at runtime)"
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=None,
        help="Dropout rate for MLP (required at runtime)"
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=None,
        help="Training epochs for MLP (required at runtime)"
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=None,
        help="Learning rate for MLP (required at runtime)"
    )
    parser.add_argument(
        "--mlp-weight-decay",
        type=float,
        default=None,
        help="Weight decay for MLP (required at runtime)"
    )
    parser.add_argument(
        "--mlp-early-stop-tol",
        type=float,
        default=None,
        help="Early stopping tolerance for MLP training convergence (required at runtime)"
    )
    parser.add_argument(
        "--mlp-input-divisor",
        type=int,
        default=None,
        help="Divisor to cap MLP hidden dim relative to input dim (required at runtime)"
    )
    parser.add_argument(
        "--mlp-early-stopping-patience",
        type=int,
        default=None,
        help="Patience epochs for MLP early stopping (required at runtime)"
    )
    parser.add_argument(
        "--mlp-gating-hidden-dim-divisor",
        type=int,
        default=None,
        help="Divisor for MLP gating network bottleneck layer (required at runtime)"
    )

    # Ostrze parameters
    parser.add_argument(
        "--ostrze-max-iter",
        type=int,
        default=None,
        help="Max iterations for logistic regression (required at runtime)"
    )
    parser.add_argument(
        "--ostrze-C",
        type=float,
        default=None,
        help="Regularization strength for logistic regression (required at runtime)"
    )

    # TECZA parameters
    parser.add_argument(
        "--tecza-num-directions",
        type=int,
        default=None,
        help="Number of directions to discover per layer (required at runtime)"
    )
    parser.add_argument(
        "--tecza-optimization-steps",
        type=int,
        default=None,
        help="Optimization steps for TECZA (required at runtime)"
    )
    parser.add_argument(
        "--tecza-learning-rate",
        type=float,
        default=None,
        help="Learning rate for TECZA (required at runtime)"
    )

    # Sensor-based method parameters come from the canonical method registry.
    # Their prefixed layer flags share the generic destinations consumed by the
    # create helpers and by MethodConfig.to_args().
    SteeringMethodRegistry.get("tetno").add_cli_arguments(parser)
    SteeringMethodRegistry.get("grom").add_cli_arguments(parser)

    # Concept Flow parameters
    parser.add_argument(
        "--nurt-num-dims",
        type=int,
        default=0,
        help="Concept subspace dimensions (0 = auto from variance, default: 0)"
    )
    parser.add_argument(
        "--nurt-variance-threshold",
        type=float,
        default=None,
        help="Cumulative variance threshold for auto dim selection (required at runtime)"
    )
    parser.add_argument(
        "--nurt-training-epochs",
        type=int,
        default=None,
        help="Training epochs for flow matching (required at runtime)"
    )
    parser.add_argument(
        "--nurt-lr",
        type=float,
        default=None,
        help="Learning rate for flow network (required at runtime)"
    )
    parser.add_argument(
        "--nurt-num-integration-steps",
        type=int,
        default=None,
        help="Euler integration steps at inference (required at runtime)"
    )
    parser.add_argument(
        "--nurt-t-max",
        type=float,
        default=None,
        help="Integration endpoint / max steering strength (required at runtime)"
    )
    parser.add_argument(
        "--nurt-hidden-dim",
        type=int,
        default=0,
        help="Velocity network hidden dim (0 = auto, default: 0)"
    )

    # Display options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show timing information"
    )
