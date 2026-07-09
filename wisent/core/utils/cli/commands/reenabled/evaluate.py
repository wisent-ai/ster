"""CLI execution for the re-enabled evaluate command.

Registered in the parser (evaluate_parser.py) but its dispatch handler had been
dropped. Loads a saved steering object, generates a steered response to a single
prompt, and scores it on two axes with a judge model. HEAVY: loads a HF causal
LM for both generation and judging, so all model imports are lazy. Mirrors the
JSON emit style of inference_config_cli.py.
"""

import json
import re

from wisent.core.utils.config_tools.constants import JSON_INDENT

_TRAIT_LOW, _TRAIT_HIGH = -1.0, 1.0
_ANSWER_LOW, _ANSWER_HIGH = 0.0, 1.0
_JUDGE_BATCH = 1
_JUDGE_TEMPERATURE = 0.0


def _extract_score(text, label, low, high):
    """First number after LABEL (else first number), clamped to [low, high]."""
    match = re.search(label + r"\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match is None:
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if match is None:
        return (low + high) / len("ab")
    return max(low, min(high, float(match.group(1))))


def execute_evaluate(args):
    """Evaluate a single prompt with a steering vector; prints a JSON verdict."""
    try:
        # steering_object.py:83
        from wisent.core.control.steering_methods.implementations.steering_object import (
            load_steering_object,
        )
        # wisent_model.py:86 (generate/set_steering_from_raw/clear_steering/apply_steering/detach)
        from wisent.core.primitives.models.core.wisent_model import WisentModel
        from wisent.core.control.generation.prompts.core.atom import ChatMessage
        from wisent.core.utils.services.judges import make_judge  # judges/base.py:30

        obj = load_steering_object(args.vector)
        model = WisentModel(args.model, device=args.device)
        messages = [[ChatMessage(role="user", content=args.prompt)]]

        # generation.py:71 applies the steering object when passed
        steered = model.generate(
            messages,
            steering_object=obj,
            steering_strength=args.steering_strength,
            max_new_tokens=args.max_new_tokens,
        )[0]
        model.detach()  # wisent_model.py:278
        baseline = model.generate(messages, max_new_tokens=args.max_new_tokens)[0]

        judge = make_judge(
            args.model,
            device=args.device,
            batch_size=_JUDGE_BATCH,
            max_new_tokens=args.max_new_tokens,
            temperature=_JUDGE_TEMPERATURE,
        )
        trait_desc = args.trait_description or args.trait
        rubric = (
            "You are grading an AI response on two independent axes. "
            "Reply with exactly two lines and nothing else.\n"
            f"TRAIT: how strongly the response expresses the trait '{trait_desc}', "
            f"from {_TRAIT_LOW} (opposite) to {_TRAIT_HIGH} (strongly present).\n"
            f"QUALITY: overall answer quality/coherence, from {_ANSWER_LOW} "
            f"(unusable) to {_ANSWER_HIGH} (excellent).\n\n"
            f"Prompt: {args.prompt}\n"
            f"Response: {steered}\n\n"
            "TRAIT: <number>\nQUALITY: <number>"
        )
        verdict = judge.complete_batch([rubric])[0]
        trait_score = _extract_score(verdict, "TRAIT", _TRAIT_LOW, _TRAIT_HIGH)
        answer_score = _extract_score(verdict, "QUALITY", _ANSWER_LOW, _ANSWER_HIGH)

        result = {
            "prompt": args.prompt,
            "trait": args.trait,
            "steering_strength": args.steering_strength,
            "response": steered,
            "trait_score": trait_score,
            "answer_score": answer_score,
            "trait_threshold": args.trait_threshold,
            "answer_threshold": args.answer_threshold,
            "passed_trait": trait_score >= args.trait_threshold,
            "passed_answer": answer_score >= args.answer_threshold,
        }
        if args.verbose:
            result["baseline_response"] = baseline
            result["judge_verdict"] = verdict
        print(json.dumps(result, indent=JSON_INDENT))
        return result
    except Exception as exc:  # noqa: BLE001 - surface JSON error like sibling handlers
        error = {"status": "error", "error": str(exc)}
        print(json.dumps(error, indent=JSON_INDENT))
        return error
