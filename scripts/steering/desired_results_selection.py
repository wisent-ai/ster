#!/usr/bin/env python3
"""Exact immutable inventory-selection contract shared by producers and workers."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

SCHEMA_VERSION = 3
PILOT_TARGET_COUNT = 4
SELECTION_KEYS = frozenset({
    "schema_version", "inventory_sha256", "target_ids", "selection_sha256",
})


class SelectionError(ValueError):
    """An inventory selection does not satisfy the immutable pilot contract."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"inventory selection is not canonical JSON data: {exc}") from exc


def _sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise SelectionError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def validate_inventory_selection(selection: Any, inventory_sha256: str) -> str:
    """Validate one exact, sorted, four-target selection and return its logical seal."""
    if not isinstance(selection, Mapping) or set(selection) != SELECTION_KEYS:
        actual = sorted(selection) if isinstance(selection, Mapping) else type(selection).__name__
        raise SelectionError(
            f"inventory selection must have exactly {sorted(SELECTION_KEYS)}; got {actual}"
        )
    if selection["schema_version"] != SCHEMA_VERSION:
        raise SelectionError(f"inventory selection schema_version must be {SCHEMA_VERSION}")
    expected_inventory_sha = _sha256(inventory_sha256, "inventory_sha256")
    observed_inventory_sha = _sha256(
        selection["inventory_sha256"], "inventory selection.inventory_sha256",
    )
    if observed_inventory_sha != expected_inventory_sha:
        raise SelectionError("inventory selection binds a different inventory")

    target_ids = selection["target_ids"]
    if (not isinstance(target_ids, list) or len(target_ids) != PILOT_TARGET_COUNT
            or any(not isinstance(target_id, str) or not target_id for target_id in target_ids)
            or len(set(target_ids)) != len(target_ids)):
        raise SelectionError(
            f"inventory selection must contain exactly {PILOT_TARGET_COUNT} unique non-empty target_ids"
        )
    if target_ids != sorted(target_ids):
        raise SelectionError("inventory selection target_ids must be sorted canonically")

    claimed = _sha256(selection["selection_sha256"], "inventory selection.selection_sha256")
    logical_payload = dict(selection)
    del logical_payload["selection_sha256"]
    observed = hashlib.sha256(_canonical_json(logical_payload)).hexdigest()
    if claimed != observed:
        raise SelectionError(
            "inventory selection.selection_sha256 does not match its logical payload"
        )
    return claimed
