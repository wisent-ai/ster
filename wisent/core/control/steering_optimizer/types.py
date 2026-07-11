"""Shared types for steering optimization."""
from __future__ import annotations

from enum import Enum


class SteeringApplicationStrategy(str, Enum):
    """How steering strength changes over generated tokens."""

    CONSTANT = "constant"
    INITIAL_ONLY = "initial_only"
    DIMINISHING = "diminishing"
    INCREASING = "increasing"
    GAUSSIAN = "gaussian"


__all__ = ["SteeringApplicationStrategy"]
