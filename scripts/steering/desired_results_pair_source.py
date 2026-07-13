#!/usr/bin/env python3
"""Exact immutable source contract for selected pair-text artifacts."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping

PAIR_SOURCE_COMMON_KEYS = frozenset({
    "repo_id", "repo_type", "revision", "path", "storage", "size",
    "content_sha256", "schema_variant",
})
PAIR_SOURCE_GIT_KEYS = PAIR_SOURCE_COMMON_KEYS | {"git_oid"}
PAIR_SOURCE_LFS_KEYS = PAIR_SOURCE_COMMON_KEYS | {"lfs_sha256"}
_HEX = frozenset("0123456789abcdef")


class PairSourceError(ValueError):
    """A pair-text source does not have an exact immutable identity."""


def _hex(value: Any, length: int, label: str) -> str:
    if (not isinstance(value, str) or len(value) != length
            or any(character not in _HEX for character in value)):
        raise PairSourceError(f"{label} must be a lowercase {length}-character hex digest")
    return value


def validate_pair_source(value: Any, label: str = "pair source") -> dict[str, Any]:
    """Validate and normalize the shared Git-blob/LFS pair-source schema."""
    if not isinstance(value, Mapping):
        raise PairSourceError(f"{label} must be an object")
    storage = value.get("storage")
    if storage == "git":
        expected_keys = PAIR_SOURCE_GIT_KEYS
    elif storage == "lfs":
        expected_keys = PAIR_SOURCE_LFS_KEYS
    else:
        raise PairSourceError(f"{label}.storage must be 'git' or 'lfs'")
    if set(value) != set(expected_keys):
        actual = sorted(str(key) for key in value)
        raise PairSourceError(
            f"{label} keys for {storage} storage must be exactly "
            f"{sorted(expected_keys)}; got {actual}"
        )

    for key in ("repo_id", "schema_variant"):
        if not isinstance(value[key], str) or not value[key]:
            raise PairSourceError(f"{label}.{key} must be a non-empty string")
    if value["repo_type"] not in {"dataset", "model", "space"}:
        raise PairSourceError(f"{label}.repo_type is invalid")
    _hex(value["revision"], 40, f"{label}.revision")

    path = value["path"]
    if (not isinstance(path, str) or not path or "\0" in path or "\\" in path
            or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise PairSourceError(f"{label}.path must be a safe non-empty POSIX path")
    PurePosixPath(path)

    if type(value["size"]) is not int or value["size"] <= 0:
        raise PairSourceError(f"{label}.size must be a positive integer")
    content_sha256 = _hex(value["content_sha256"], 64, f"{label}.content_sha256")
    if storage == "git":
        _hex(value["git_oid"], 40, f"{label}.git_oid")
    else:
        lfs_sha256 = _hex(value["lfs_sha256"], 64, f"{label}.lfs_sha256")
        if content_sha256 != lfs_sha256:
            raise PairSourceError(f"{label}.content_sha256 must equal its LFS SHA-256")
    return dict(value)
