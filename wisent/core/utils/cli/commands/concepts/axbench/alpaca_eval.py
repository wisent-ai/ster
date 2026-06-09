"""Alpaca-Eval instruction sampling for the AxBench steering protocol.

AxBench samples N instructions per concept (reference: 10, seed 42) from
Alpaca-Eval and partitions them into two equal sets: one for selecting the
best steering factor, the other held out for the reported evaluation.
"""

import random
from typing import List, Tuple

import datasets

__all__ = ["load_alpaca_instructions", "split_select_eval"]

_ALPACA_EVAL_SOURCE = "tatsu-lab/alpaca_eval"
_ALPACA_EVAL_CONFIG = "alpaca_eval"
_ALPACA_EVAL_SPLIT = "eval"


def load_alpaca_instructions(n: int, seed: int) -> List[str]:
    """Deterministically sample n instructions from Alpaca-Eval."""
    dataset = datasets.load_dataset(
        _ALPACA_EVAL_SOURCE,
        _ALPACA_EVAL_CONFIG,
        split=_ALPACA_EVAL_SPLIT,
        trust_remote_code=True,
    )
    instructions = [str(row["instruction"]) for row in dataset]
    if n > len(instructions):
        raise ValueError(
            f"Requested {n} instructions but {_ALPACA_EVAL_SOURCE} has only "
            f"{len(instructions)}."
        )
    rng = random.Random(seed)
    return rng.sample(instructions, n)


def split_select_eval(instructions: List[str]) -> Tuple[List[str], List[str]]:
    """Partition instructions into equal select/eval halves (AxBench split 0.5)."""
    if len(instructions) < 2:
        raise ValueError(
            f"Need at least 2 instructions to split into select/eval halves, "
            f"got {len(instructions)}."
        )
    half = len(instructions) // 2
    return instructions[:half], instructions[half:]
