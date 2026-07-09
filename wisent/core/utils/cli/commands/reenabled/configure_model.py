"""CLI execution for the 're-enabled' configure-model command.

Registered in the argparse parser (configure_model_parser.py) but its dispatch
handler had been dropped. Mirrors the emit/print style of
wisent/core/utils/cli/analysis/analysis/config/inference_config_cli.py.
"""

import json

from wisent.core.utils.config_tools.constants import JSON_INDENT, SEPARATOR_WIDTH_MEDIUM


def execute_configure_model(args):
    """Execute the configure-model command.

    Behaviour:
      * If the model already has a stored user config and ``--force`` was not
        passed, print the existing configuration (pretty JSON) plus a note that
        ``--force`` re-runs configuration.
      * Otherwise interactively prompt for the configuration, save it, and print
        the saved configuration (pretty JSON).

    ``args`` is an argparse.Namespace with dests from
    configure_model_parser.py:6-7 -> ``model`` (positional str), ``force`` (bool).
    """
    model = args.model
    force = getattr(args, "force", False)

    try:
        # Lazy import to keep CLI startup light and avoid import chains. The
        # module-global instance is the public entrypoint.
        # user_model_configs: wisent/core/primitives/models/extended/config/user_model_config.py:160
        from wisent.core.primitives.models.extended.config.user_model_config import (
            user_model_configs,
        )

        # has_config: user_model_config.py:48 ; get_config: user_model_config.py:52
        if user_model_configs.has_config(model) and not force:
            config = user_model_configs.get_config(model)
            print(f"Configuration for '{model}':")
            print("-" * SEPARATOR_WIDTH_MEDIUM)
            print(json.dumps(config, indent=JSON_INDENT))
            print("\nModel already configured. Use --force to reconfigure it.")
        else:
            # prompt_and_save_config: user_model_config.py:84 (interactive stdin;
            # prompts, then persists via save_config -> user_model_config.py:56)
            config = user_model_configs.prompt_and_save_config(model)
            print(f"\nSaved configuration for '{model}':")
            print("-" * SEPARATOR_WIDTH_MEDIUM)
            print(json.dumps(config, indent=JSON_INDENT))
    except (KeyboardInterrupt, EOFError):
        # Interactive prompt aborted (no/closed stdin): report, do not crash.
        print(
            json.dumps(
                {
                    "command": "configure-model",
                    "model": model,
                    "status": "aborted",
                    "error": "configuration input was interrupted",
                },
                indent=JSON_INDENT,
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure as JSON
        print(
            json.dumps(
                {
                    "command": "configure-model",
                    "model": model,
                    "status": "error",
                    "error": str(exc),
                },
                indent=JSON_INDENT,
            )
        )
