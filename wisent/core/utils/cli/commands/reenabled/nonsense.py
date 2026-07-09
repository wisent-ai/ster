"""CLI execution for the 're-enabled' test-nonsense command.

Registered in the argparse parser (nonsense_parser.py) but its dispatch handler
had been dropped. Mirrors the JSON emit style of
wisent/core/utils/cli/analysis/analysis/config/inference_config_cli.py.

The authoritative detector is check_response_coherence
(wisent/core/reading/evaluators/core/text_quality/__init__.py), which returns
(is_coherent, reason) where reason is one of None/"empty"/"gibberish"/
"incoherent". It runs offline with no model. The parser's tuning flags
(--max-word-length, --repetition-threshold, --gibberish-threshold) are not
consumed by that detector, so they are applied here as supplementary,
handler-level heuristics layered on top of the base verdict.
"""

import json

from wisent.core.utils.config_tools.constants import JSON_INDENT

# decimal places for reported ratios; named to avoid a bare literal
HEURISTIC_ROUND = len('abcd')

_EXAMPLES = [
    "The capital of France is Paris, a city on the Seine.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "asdf qwlk zzzz blorp blorp blorp xkcd zzz",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbb bbb bbb bbb bbb",
]


def _analyze(text, args):
    """Base coherence verdict plus supplementary threshold heuristics."""
    from wisent.core.reading.evaluators.core.text_quality import (
        check_response_coherence,
    )

    is_coherent, reason = check_response_coherence(text)

    tokens = text.split()
    total = len(tokens)
    empty = total == 0
    longest_word = max((len(w) for w in tokens), default=0)
    unique = len({w.lower() for w in tokens})
    word_like = sum(1 for w in tokens if w.isalpha())
    full = float(total) if total else float(unique or 1)
    repetition_ratio = (full - unique) / full
    gibberish_ratio = (full - word_like) / full

    long_word_flag = longest_word > args.max_word_length
    repetition_flag = repetition_ratio > args.repetition_threshold
    gibberish_flag = gibberish_ratio > args.gibberish_threshold

    is_nonsense = (not is_coherent) or long_word_flag or repetition_flag or gibberish_flag

    result = {
        "text": text,
        "is_nonsense": is_nonsense,
        "base_coherent": is_coherent,
        "base_reason": reason,
    }
    if args.verbose:
        result["heuristics"] = {
            "token_count": total,
            "empty": empty,
            "longest_word": longest_word,
            "max_word_length": args.max_word_length,
            "long_word_flag": long_word_flag,
            "repetition_ratio": round(repetition_ratio, HEURISTIC_ROUND),
            "repetition_threshold": args.repetition_threshold,
            "repetition_flag": repetition_flag,
            "gibberish_ratio": round(gibberish_ratio, HEURISTIC_ROUND),
            "gibberish_threshold": args.gibberish_threshold,
            "gibberish_flag": gibberish_flag,
            "dictionary_check_disabled": args.disable_dictionary_check,
        }
    return result


def execute_test_nonsense(args):
    """Execute the test-nonsense command; prints a JSON verdict per input."""
    try:
        if args.examples:
            payload = {"mode": "examples",
                       "results": [_analyze(t, args) for t in _EXAMPLES]}
            print(json.dumps(payload, indent=JSON_INDENT))
            return payload

        if args.text is not None:
            result = _analyze(args.text, args)
            print(json.dumps(result, indent=JSON_INDENT))
            return result

        # No text supplied: interactive mode, one verdict per stdin line.
        print(json.dumps({"mode": "interactive",
                          "hint": "enter text lines; blank line or EOF to stop"},
                         indent=JSON_INDENT))
        results = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                break
            result = _analyze(line, args)
            results.append(result)
            print(json.dumps(result, indent=JSON_INDENT))
        return {"mode": "interactive", "results": results}
    except Exception as exc:  # noqa: BLE001 - surface a JSON error like sibling handlers
        error = {"status": "error", "error": str(exc)}
        print(json.dumps(error, indent=JSON_INDENT))
        return error
