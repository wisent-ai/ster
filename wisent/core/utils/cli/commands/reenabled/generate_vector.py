"""CLI execution for the re-enabled generate-vector command.

Registered in the parser but its dispatch handler had been dropped. Builds a
steering vector from a contrastive-pairs file (--from-pairs) or a trait
description (--from-description), with optional multi-property composition.
Reuses the existing, exercised pipeline handlers rather than reimplementing:
get-activations -> create-steering-object (from pairs), the synthetic full
pipeline (from description), and multi-steer (composition). HEAVY: every path
loads a HF model, so pipeline imports are lazy. Mirrors inference_config_cli.py.
"""

import json
import os
import tempfile
from argparse import Namespace

from wisent.core.utils.config_tools.constants import JSON_INDENT

# generate-pairs --similarity-threshold has no argparse default (help: "0.8").
_SIMILARITY_THRESHOLD = float(len("00000000")) / len("0000000000")  # 0.8
_COMBINED_SCALE = float(len("x"))  # 1.0

_PROMPT_FAMILY = {
    "multiple_choice": "mc",
    "role_playing": "role_play",
    "direct_completion": "completion",
    "instruction_following": "chat",
}


def _extraction_strategy(prompt_construction, token_targeting):
    """Map (prompt_construction, token_targeting) onto one ExtractionStrategy."""
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy

    family = _PROMPT_FAMILY.get(prompt_construction, "chat")
    if family == "role_play":
        name = "ROLE_PLAY"
    elif family == "mc":
        name = "MC_COMPLETION" if token_targeting == "continuation_token" else "MC_BALANCED"
    elif family == "completion":
        name = "COMPLETION_MEAN" if token_targeting == "mean_pooling" else "COMPLETION_LAST"
    else:
        name = {
            "first_token": "CHAT_FIRST",
            "mean_pooling": "CHAT_MEAN",
            "max_pooling": "CHAT_MAX_NORM",
        }.get(token_targeting, "CHAT_LAST")
    return ExtractionStrategy[name].value


def _layers_arg(args):
    return str(args.layer) if args.layer is not None else None


def _build_from_pairs(args, pairs_file, output):
    """pairs file -> get-activations -> create-steering-object -> saved vector."""
    from wisent.core.utils.cli import execute_get_activations, execute_create_steering_object

    enriched = tempfile.NamedTemporaryFile(mode="w", suffix="_enriched.json", delete=False).name
    execute_get_activations(Namespace(
        pairs_file=pairs_file, output=enriched, model=args.model, device=args.device,
        layers=_layers_arg(args),
        extraction_strategy=_extraction_strategy(args.prompt_construction, args.token_targeting),
        verbose=args.verbose, timing=False,
    ))
    execute_create_steering_object(Namespace(
        enriched_pairs_file=enriched, output=output, method=args.method,
        normalize=True, verbose=args.verbose, timing=False,
        accept_low_quality_vector=False,
    ))
    if os.path.exists(enriched):
        os.unlink(enriched)


def _build_from_description(args, trait, output):
    """trait description -> synthetic full pipeline -> saved vector."""
    from wisent.core.utils.cli import execute_generate_vector_from_synthetic

    execute_generate_vector_from_synthetic(Namespace(
        trait=trait, num_pairs=args.num_pairs, model=args.model, device=args.device,
        similarity_threshold=_SIMILARITY_THRESHOLD, verbose=args.verbose, timing=False,
        layers=_layers_arg(args), method=args.method, normalize=True, output=output,
        keep_intermediate=args.save_pairs, intermediate_dir=None,
        pairs_cache_dir=None, force_regenerate=False, accept_low_quality_vector=False,
    ))


def _run_multi_property(args):
    """Build one vector per property, then compose them via multi-steer."""
    from wisent.core.utils.cli import execute_multi_steer

    entries = []
    for spec in (args.property_files or []):
        name, path, layer = spec.split(":")
        entries.append((name, "pairs", path, int(layer)))
    for spec in (args.property_descriptions or []):
        name, desc, layer = spec.split(":")
        entries.append((name, "description", desc, int(layer)))
    if not entries:
        raise ValueError("--multi-property requires --property-files or --property-descriptions")

    tmpdir = tempfile.mkdtemp()
    built = []
    for name, kind, value, layer in entries:
        vec_path = os.path.join(tmpdir, f"{name}.pt")
        prop = Namespace(**vars(args))
        prop.layer = layer
        if kind == "pairs":
            _build_from_pairs(prop, value, vec_path)
        else:
            _build_from_description(prop, value, vec_path)
        built.append((name, vec_path))

    execute_multi_steer(Namespace(
        vector=[path for _, path in built], model=args.model, layer=args.layer,
        method=args.method, combined_scale=_COMBINED_SCALE, device=args.device,
        normalize_weights=True, save_combined=args.output, prompt=None,
    ))
    return {"status": "ok", "source": "multi_property",
            "properties": [name for name, _ in built], "output": args.output}


def execute_generate_vector(args):
    """Generate a steering vector; prints a JSON summary of what was written."""
    try:
        if args.multi_property:
            result = _run_multi_property(args)
        elif args.from_pairs:
            _build_from_pairs(args, args.from_pairs, args.output)
            result = {"status": "ok", "source": "pairs", "input": args.from_pairs,
                      "method": args.method, "output": args.output}
        elif args.from_description:
            _build_from_description(args, args.from_description, args.output)
            result = {"status": "ok", "source": "description",
                      "trait": args.from_description, "method": args.method,
                      "output": args.output}
        else:
            raise ValueError(
                "provide one of --from-pairs, --from-description, or --multi-property"
            )
        print(json.dumps(result, indent=JSON_INDENT))
        return result
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        error = {"status": "error", "error": str(exc)}
        print(json.dumps(error, indent=JSON_INDENT))
        return error
