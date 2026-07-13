"""Canonical pair identity validation for raw activation artifacts."""

from __future__ import annotations

from typing import Union


PairId = Union[int, str]


def validate_pair_id(value: object) -> PairId:
    """Return a canonical scalar pair ID without coercing its type or value."""
    if type(value) is int:
        if value < 0:
            raise ValueError("pair_id integer must be nonnegative")
        return value
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("pair_id must be a nonnegative integer or non-empty string")
