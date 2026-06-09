"""Parser setup for the 'axbench' command.

Defaults reproduce the published AxBench protocol (arXiv 2501.17148,
reference sweep config steering_vec.yaml): 14 steering factors, 10
Alpaca-Eval instructions per concept (seed 42, split 0.5 select/eval),
128-token generations, judge ratings on a 0-2 scale.
"""

# Reference steering-factor grid from the AxBench sweep configs.
AXBENCH_REFERENCE_FACTORS = "0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0,5.0"
AXBENCH_N_INSTRUCTIONS = 10
AXBENCH_SEED = 42
AXBENCH_MAX_NEW_TOKENS = 128
AXBENCH_GENERATION_TEMPERATURE = 1.0
AXBENCH_JUDGE_BATCH_SIZE = 16
AXBENCH_JUDGE_MAX_NEW_TOKENS = 512
AXBENCH_JUDGE_TEMPERATURE = 0.0
AXBENCH_IMBALANCED_NEGATIVES = 3600


def setup_axbench_parser(parser):
    """Set up the axbench command parser."""
    parser.add_argument("--model", type=str, required=True,
                        help="Subject model to steer/probe (HuggingFace id)")
    parser.add_argument("--variant", type=str, default="concept500",
                        choices=["concept500", "concept16k", "concept16k_v2"],
                        help="AxBench dataset variant (default: concept500, the paper's eval set)")
    parser.add_argument("--concept-set", type=str, default="2b/l10",
                        choices=["2b/l10", "2b/l20", "9b/l20", "9b/l31"],
                        help="Concept set by GemmaScope subject model/layer (default: 2b/l10)")
    parser.add_argument("--axbench-action", type=str, required=True,
                        choices=["steer", "detect"],
                        help="steer: judge-scored steering protocol; detect: AUROC/F1 concept detection")
    parser.add_argument("--concept-id", type=int, default=None,
                        help="Run a single concept by id")
    parser.add_argument("--all-concepts", action="store_true",
                        help="Run every concept in the variant")
    parser.add_argument("--max-concepts", type=int, default=None,
                        help="Cap the number of concepts when using --all-concepts")
    parser.add_argument("--layer", type=int, default=None,
                        help="1-indexed layer for extraction/steering (default: model midpoint)")
    parser.add_argument("--method", type=str, default="caa",
                        help="Wisent steering method (default: caa, equivalent to the AxBench DiffMean baseline)")
    parser.add_argument("--method-params-file", type=str, default=None,
                        help="JSON file of method-specific hyperparameters passed to "
                             "create-steering-vector (e.g. grom_*, mlp_*, nurt_*); required "
                             "for methods whose trainer demands explicit hyperparameters")
    parser.add_argument("--factors", type=str, default=AXBENCH_REFERENCE_FACTORS,
                        help="Comma-separated steering factors (default: the 14 AxBench reference factors)")
    parser.add_argument("--n-instructions", type=int, default=AXBENCH_N_INSTRUCTIONS,
                        help="Alpaca-Eval instructions per concept (AxBench reference: 10)")
    parser.add_argument("--seed", type=int, default=AXBENCH_SEED,
                        help="Sampling seed (AxBench reference: 42)")
    parser.add_argument("--max-new-tokens", type=int, default=AXBENCH_MAX_NEW_TOKENS,
                        help="Generation budget per instruction (AxBench reference: 128)")
    parser.add_argument("--temperature", type=float, default=AXBENCH_GENERATION_TEMPERATURE,
                        help="Generation temperature (AxBench reference: 1.0; 0 = greedy)")
    parser.add_argument("--use-hard-negatives", action="store_true",
                        help="Include the concept's hard-negative rows in the training pair pool")
    parser.add_argument("--pair-limit", type=int, default=None,
                        help="Cap on training pairs per concept (default: all available)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge for steer: 'openai:gpt-4o-mini' (AxBench reference, needs "
                             "OPENAI_API_KEY) or a HuggingFace model id for a local judge")
    parser.add_argument("--judge-batch-size", type=int, default=AXBENCH_JUDGE_BATCH_SIZE,
                        help="Concurrent judge requests / local judge batch (AxBench reference: 16)")
    parser.add_argument("--judge-max-new-tokens", type=int, default=AXBENCH_JUDGE_MAX_NEW_TOKENS,
                        help="Completion budget per judge rubric prompt")
    parser.add_argument("--judge-temperature", type=float, default=AXBENCH_JUDGE_TEMPERATURE,
                        help="Judge sampling temperature (0 = deterministic ratings)")
    parser.add_argument("--imbalanced-negatives", type=int, default=AXBENCH_IMBALANCED_NEGATIVES,
                        help="Negatives in the imbalanced detection setting (AxBench reference: 3600)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file for the summary")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for pairs/activations/steering objects "
                             "(default: axbench_work next to --output)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for the subject model and local judge")
