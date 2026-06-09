"""Parser setup for the 'evaluate-responses' command."""

from wisent.core.utils.config_tools.parser_arguments.concepts.axbench_parser import (
    AXBENCH_JUDGE_BATCH_SIZE,
    AXBENCH_JUDGE_MAX_NEW_TOKENS,
    AXBENCH_JUDGE_TEMPERATURE,
)


def setup_evaluate_responses_parser(parser):
    """Set up the evaluate-responses command parser."""
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model name to load for evaluation (e.g. meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with generated responses")
    parser.add_argument("--baseline", type=str, help="Baseline responses JSON file (for personalization comparison)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for evaluation results")
    parser.add_argument("--task", type=str, help="Task name (optional, overrides task from input JSON)")
    parser.add_argument("--trait", type=str, help="Personality trait to evaluate (optional, for personalization tasks)")
    parser.add_argument("--trait-description", type=str, help="Description of the personality trait")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge for axbench tasks: 'openai:gpt-4o-mini' (AxBench reference, "
                             "needs OPENAI_API_KEY) or a HuggingFace model id for a local judge")
    parser.add_argument("--judge-batch-size", type=int, default=AXBENCH_JUDGE_BATCH_SIZE,
                        help="Concurrent judge requests / local judge batch (AxBench reference: 16)")
    parser.add_argument("--judge-max-new-tokens", type=int, default=AXBENCH_JUDGE_MAX_NEW_TOKENS,
                        help="Completion budget per judge rubric prompt")
    parser.add_argument("--judge-temperature", type=float, default=AXBENCH_JUDGE_TEMPERATURE,
                        help="Judge sampling temperature (0 = deterministic ratings)")
    parser.add_argument("--concept-label", type=str, default=None,
                        help="AxBench concept label override when responses lack concept metadata")
    parser.add_argument("--subprocess-timeout", type=int, required=True,
                        help="Timeout in seconds for subprocess execution (e.g. Docker code evaluation)")
    parser.add_argument("--personalization-good-threshold", type=int, required=True,
                        help="Threshold score for good personalization")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--fast-diversity-seed", type=int, required=True,
        help="Seed for fast diversity computation"
    )
    parser.add_argument(
        "--diversity-max-sample-size", type=int, required=True,
        help="Maximum sample size for diversity computation"
    )
    parser.add_argument(
        "--min-sentence-length", type=int, required=True,
        help="Minimum sentence length for coherence evaluation"
    )
    parser.add_argument(
        "--nonsense-min-tokens", type=int, required=True,
        help="Minimum token count for nonsense word detection"
    )
    parser.add_argument(
        "--quality-min-response-length", type=int, required=True,
        help="Minimum response length for quality scoring"
    )
    parser.add_argument(
        "--quality-repetition-ratio-threshold", type=float, required=True,
        help="Threshold for repetitive token ratio penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-threshold", type=int, required=True,
        help="Threshold for repeated bigram count penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for repeated bigrams"
    )
    parser.add_argument(
        "--quality-special-char-ratio-threshold", type=float, required=True,
        help="Threshold for special character ratio penalty"
    )
    parser.add_argument(
        "--quality-special-char-penalty", type=float, required=True,
        help="Penalty multiplier for excessive special characters"
    )
    parser.add_argument(
        "--quality-char-repeat-count", type=int, required=True,
        help="Minimum consecutive character repeats to trigger penalty"
    )
    parser.add_argument(
        "--quality-char-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for character repetition"
    )
    parser.add_argument(
        "--f1-threshold", type=float, required=True,
        help="F1 score threshold for token-level evaluator"
    )
    parser.add_argument(
        "--generation-embedding-weight", type=float, required=True,
        help="Weight for embedding similarity in generation evaluator"
    )
    parser.add_argument(
        "--generation-nli-weight", type=float, required=True,
        help="Weight for NLI entailment in generation evaluator"
    )
    parser.add_argument(
        "--personalization-difference-weight", type=float, required=True,
        help="Weight for difference score in personalization evaluation"
    )
    parser.add_argument(
        "--personalization-quality-weight", type=float, required=True,
        help="Weight for quality score in personalization evaluation"
    )
    parser.add_argument(
        "--personalization-alignment-weight", type=float, required=True,
        help="Weight for alignment score in personalization evaluation"
    )
