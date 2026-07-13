#!/usr/bin/env python3
"""Run activation preflight from an exact GCS descriptor and publish its immutable bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


def _load(name: str) -> Any:
    path = Path(__file__).with_name(name + ".py")
    repo_root = str(path.resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location("_" + name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load("desired_results_activation_preflight")
RUN_V3 = _load("desired_results_run_v3")
execute_remote = PREFLIGHT.execute_remote


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("descriptor", "submission"):
        parser.add_argument(f"--{prefix}-uri", required=True)
        parser.add_argument(f"--{prefix}-generation", required=True)
        parser.add_argument(f"--{prefix}-sha256", required=True)
        parser.add_argument(f"--{prefix}-size", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--terminal-receipt-uri", required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.workers < 1 or args.workers > 32:
            raise PREFLIGHT.PreflightError("--workers must be between 1 and 32")
        descriptor_ref = {
            key: getattr(args, "descriptor_" + key)
            for key in ("uri", "generation", "sha256", "size")
        }
        submission_ref = {
            key: getattr(args, "submission_" + key)
            for key in ("uri", "generation", "sha256", "size")
        }
        receipt = execute_remote(
            descriptor_ref, args.output_prefix, args.terminal_receipt_uri,
            node_id=args.node_id, submission_ref=submission_ref,
            workers=args.workers, store=RUN_V3.GCSStore(),
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        return 0
    except (PREFLIGHT.PreflightError, RUN_V3.RunV3Error, OSError, ValueError, TypeError) as exc:
        print(f"activation preflight worker failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
