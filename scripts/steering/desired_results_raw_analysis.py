#!/usr/bin/env python3
"""Analyze immutable, packed raw activation trajectories without split leakage.

This entry point deliberately accepts only the desired-results raw route contract.  It
is not a compatibility reader for Wisent's historical ``raw_mode`` or ``.pt`` cache
formats: those formats cannot prove token boundaries, masks, pair identities, or
immutable source provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

import torch
from safetensors import safe_open
from safetensors.torch import load_file

try:
    from .desired_results_execution_contract import (
        artifact_ref,
        canonical_json,
        canonical_sha256,
        validate_artifact_binding,
        validate_artifact_ref,
    )
    from .desired_results_target import STRATEGIES, validate_target_manifest
except ImportError:  # pragma: no cover - direct script execution
    from desired_results_execution_contract import (  # type: ignore
        artifact_ref,
        canonical_json,
        canonical_sha256,
        validate_artifact_binding,
        validate_artifact_ref,
    )
    from desired_results_target import STRATEGIES, validate_target_manifest  # type: ignore


COMPLETION_KEYS = frozenset({
    "schema_version", "complete", "kind", "target", "revisions", "support",
    "target_manifest_ref", "artifact", "manifest_sha256",
})
TARGET_KEYS = frozenset({
    "target_id", "model", "model_slug", "benchmark", "strategy", "layer",
    "layer_count",
})
REVISION_KEYS = frozenset({"model", "tokenizer", "activation", "code", "runtime"})
SUPPORT_KEYS = frozenset({"pair_id", "stable_id", "split"})
ARTIFACT_KEYS = frozenset({"format", "content_ref", "source_route_ref"})
TENSOR_NAMES = frozenset({
    "positive_activations", "negative_activations", "positive_token_ids",
    "negative_token_ids", "positive_attention_mask", "negative_attention_mask",
})
METADATA_NAMES = frozenset({
    "pair_ids", "stable_ids", "positive_lengths", "negative_lengths",
    "positive_prompt_lengths", "negative_prompt_lengths",
    "positive_answer_onsets", "negative_answer_onsets",
})
SPLITS = frozenset({"train", "validation", "test"})
KIND = "desired_results_raw_activation_route"
FORMAT = "packed-ragged-safetensors-v1"
HEX64 = frozenset("0123456789abcdef")


class RawAnalysisError(RuntimeError):
    """The raw route or its analysis violates the immutable analysis contract."""


@dataclass(frozen=True)
class ContentRef:
    uri: str
    generation: str
    size: int
    size_text: str
    sha256: str

    def json(self) -> dict[str, str]:
        return artifact_ref(self.uri, self.generation, self.size_text, self.sha256)


@dataclass
class Route:
    completion_path: Path
    completion_sha256: str
    completion: Mapping[str, Any]
    target_manifest: Mapping[str, Any]
    artifact_ref: ContentRef
    artifact_path: Path
    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, list[Any]]
    direction: torch.Tensor | None = None
    direction_sha256: str | None = None
    direction_fit_pair_ids: tuple[int, ...] = ()
    direction_fit_stable_ids: tuple[str, ...] = ()

    @property
    def target(self) -> Mapping[str, Any]:
        return self.completion["target"]

    @property
    def support(self) -> Sequence[Mapping[str, Any]]:
        return self.completion["support"]


@dataclass(frozen=True)
class PairSlice:
    row: Mapping[str, Any]
    positive: slice
    negative: slice
    positive_length: int
    negative_length: int
    positive_prompt_length: int
    negative_prompt_length: int
    positive_answer_onset: int
    negative_answer_onset: int


def _exact(value: Any, keys: frozenset[str] | set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise RawAnalysisError(f"{label} keys must be exactly {sorted(keys)}; got {actual}")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RawAnalysisError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise RawAnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RawAnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _read_json(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawAnalysisError(f"cannot read {label} {path}: {exc}") from exc
    return value


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise RawAnalysisError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest(), size


def _content_ref(value: Any, label: str) -> ContentRef:
    try:
        ref = validate_artifact_ref(value, label)
    except Exception as exc:
        raise RawAnalysisError(f"invalid {label}: {exc}") from exc
    return ContentRef(
        ref["uri"], ref["generation"], int(ref["size"]), ref["size"], ref["sha256"],
    )


def _safe_relative_uri(uri: str, label: str) -> Path:
    # Check the literal spelling before PurePath can normalize away evidence.
    if "\\" in uri:
        raise RawAnalysisError(f"{label} contains a backslash")
    pieces = uri.split("/")
    if uri.startswith("/") or any(piece in {"", ".", ".."} for piece in pieces):
        raise RawAnalysisError(f"{label} must be a literal safe relative path")
    decoded = [unquote(piece) for piece in pieces]
    if any(piece in {"", ".", ".."} or "/" in piece or "\\" in piece for piece in decoded):
        raise RawAnalysisError(f"{label} contains an encoded unsafe path component")
    pure = PurePosixPath(*decoded)
    if pure.is_absolute():
        raise RawAnalysisError(f"{label} must not be absolute")
    return Path(*pure.parts)


def _resolve_hf(uri: str, generation: str) -> Path:
    # Canonical URI: hf://datasets/<owner>/<repo>@<revision>/<repo-path>.
    parsed = urlsplit(uri)
    if f"{parsed.scheme}://{parsed.netloc}{parsed.path}" != uri:
        raise RawAnalysisError("HF content URI must not contain a query or fragment")
    if (parsed.scheme, parsed.netloc) != ("hf", "datasets"):
        raise RawAnalysisError("HF content URI must be hf://datasets/<repo>@<revision>/<path>")
    literal = parsed.path.lstrip("/")
    if "@" not in literal:
        raise RawAnalysisError("HF content URI must pin a revision with '@'")
    repo_part, tail = literal.rsplit("@", 1)
    if "/" not in tail:
        raise RawAnalysisError("HF content URI is missing its pinned artifact path")
    revision, filename = tail.split("/", 1)
    repo_parts = repo_part.split("/")
    if len(repo_parts) != 2 or any(part in {"", ".", ".."} for part in repo_parts):
        raise RawAnalysisError("HF content URI repository must be exactly owner/name")
    if not revision or revision != generation:
        raise RawAnalysisError("HF content URI revision must equal immutable ref generation")
    safe_filename = _safe_relative_uri(filename, "HF artifact path").as_posix()
    try:
        from huggingface_hub import hf_hub_download
        materialized = hf_hub_download(
            repo_id="/".join(repo_parts), repo_type="dataset", revision=revision,
            filename=safe_filename, local_files_only=True,
        )
    except Exception as exc:
        raise RawAnalysisError(f"pinned HF artifact is not present in the local cache: {uri}") from exc
    return Path(materialized).resolve()


def _resolve_ref(ref: ContentRef, cache_dir: Path, label: str) -> Path:
    if ref.uri.startswith("hf://"):
        path = _resolve_hf(ref.uri, ref.generation)
    else:
        if "://" in ref.uri:
            raise RawAnalysisError(f"{label}.uri uses an unsupported scheme")
        relative = _safe_relative_uri(ref.uri, f"{label}.uri")
        root = cache_dir.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:  # defense in depth for symlinked parents
            raise RawAnalysisError(f"{label}.uri escapes cache_dir") from exc
    if not path.is_file():
        raise RawAnalysisError(f"{label} does not resolve to a local file: {ref.uri}")
    digest, size = _digest_file(path)
    if digest != ref.sha256 or size != ref.size:
        raise RawAnalysisError(f"{label} bytes do not match exact sha256/size")
    return path


def _same_ref(left: Any, right: Any, label: str) -> None:
    first = _content_ref(left, f"{label}.left")
    second = _content_ref(right, f"{label}.right")
    if first.json() != second.json():
        raise RawAnalysisError(f"{label} immutable references differ")


def _route_manifest_ref(manifest: Mapping[str, Any], strategy: str, layer: int) -> Mapping[str, Any]:
    matches = [route for route in manifest["activation"]["routes"]
               if route["strategy"] == strategy and route["layer"] == layer]
    if len(matches) != 1:
        raise RawAnalysisError("TargetManifest does not contain exactly one matching source route")
    return matches[0]["completion_ref"]


def _validate_support(completion: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    support = completion["support"]
    if not isinstance(support, list) or not support:
        raise RawAnalysisError("completion.support must be a non-empty list")
    seen_pairs: set[int] = set()
    seen_stable: set[str] = set()
    normalized: list[tuple[int, str, str]] = []
    for index, raw in enumerate(support):
        row = _exact(raw, SUPPORT_KEYS, f"completion.support[{index}]")
        pair_id = _integer(row["pair_id"], f"completion.support[{index}].pair_id")
        stable_id = _nonempty_string(row["stable_id"], f"completion.support[{index}].stable_id")
        split = row["split"]
        if split not in SPLITS:
            raise RawAnalysisError(f"completion.support[{index}].split is invalid")
        if pair_id in seen_pairs or stable_id in seen_stable:
            raise RawAnalysisError("completion support pair_id/stable_id identities must each be unique")
        seen_pairs.add(pair_id)
        seen_stable.add(stable_id)
        normalized.append((pair_id, stable_id, split))
    expected: list[tuple[int, str, str]] = []
    for split in ("train", "validation", "test"):
        for row in manifest["support"]["splits"][split]:
            expected.append((row["pair_id"], row["stable_id"], split))
    if normalized != expected:
        raise RawAnalysisError("completion support must exactly match ordered TargetManifest support and splits")


def _validate_completion(path: Path, cache_dir: Path) -> Route:
    value = _read_json(path, "completion manifest")
    root = _exact(value, COMPLETION_KEYS, "completion")
    if root["schema_version"] != 2 or root["complete"] is not True or root["kind"] != KIND:
        raise RawAnalysisError("completion must be a complete desired-results raw route schema v2")
    claimed = _sha(root["manifest_sha256"], "completion.manifest_sha256")
    unhashed = dict(root)
    del unhashed["manifest_sha256"]
    if claimed != canonical_sha256(unhashed):
        raise RawAnalysisError("completion.manifest_sha256 does not bind its canonical payload")

    target = _exact(root["target"], TARGET_KEYS, "completion.target")
    for name in ("target_id", "model", "model_slug", "benchmark"):
        _nonempty_string(target[name], f"completion.target.{name}")
    if target["strategy"] not in STRATEGIES:
        raise RawAnalysisError("completion target strategy is not in the frozen seven-strategy set")
    layer = _integer(target["layer"], "completion.target.layer", 1)
    layer_count = _integer(target["layer_count"], "completion.target.layer_count", 1)
    if layer > layer_count:
        raise RawAnalysisError("completion target layer exceeds layer_count")

    revisions = _exact(root["revisions"], REVISION_KEYS, "completion.revisions")
    for name in REVISION_KEYS:
        _nonempty_string(revisions[name], f"completion.revisions.{name}")

    target_ref = _content_ref(root["target_manifest_ref"], "completion.target_manifest_ref")
    target_path = _resolve_ref(target_ref, cache_dir, "completion.target_manifest_ref")
    manifest = _read_json(target_path, "TargetManifest")
    if not isinstance(manifest, Mapping):
        raise RawAnalysisError("TargetManifest must be an object")
    try:
        validate_target_manifest(manifest)
    except Exception as exc:
        raise RawAnalysisError(f"invalid TargetManifest: {exc}") from exc
    try:
        validate_artifact_binding(root["target_manifest_ref"], manifest,
                                  "completion.target_manifest_ref")
    except Exception as exc:
        raise RawAnalysisError(f"TargetManifest ArtifactRef is not canonically content-bound: {exc}") from exc
    manifest_target = manifest["target"]
    expected_target = {
        "target_id": manifest_target["target_id"], "model": manifest_target["model_name"],
        "model_slug": manifest_target["model_slug"], "benchmark": manifest_target["benchmark"],
    }
    if any(target[name] != expected for name, expected in expected_target.items()):
        raise RawAnalysisError("completion target identity differs from TargetManifest")
    if target["layer_count"] != manifest["activation"]["layer_count"]:
        raise RawAnalysisError("completion layer_count differs from TargetManifest")
    manifest_revisions = manifest["revisions"]
    for completion_name, manifest_name in (
        ("model", "model_revision"), ("tokenizer", "tokenizer_revision"),
        ("activation", "activation_revision"),
    ):
        if revisions[completion_name] != manifest_revisions[manifest_name]:
            raise RawAnalysisError(
                f"completion {completion_name} revision differs from TargetManifest"
            )
    _validate_support(root, manifest)

    artifact = _exact(root["artifact"], ARTIFACT_KEYS, "completion.artifact")
    if artifact["format"] != FORMAT:
        raise RawAnalysisError(f"completion artifact format must be {FORMAT!r}")
    source_ref = _route_manifest_ref(manifest, target["strategy"], layer)
    _same_ref(artifact["source_route_ref"], source_ref, "completion.artifact.source_route_ref")
    artifact_ref = _content_ref(artifact["content_ref"], "completion.artifact.content_ref")
    artifact_path = _resolve_ref(artifact_ref, cache_dir, "completion.artifact.content_ref")
    tensors, metadata = _load_packed(artifact_path, root["support"])
    return Route(path.resolve(), claimed, root, manifest, artifact_ref, artifact_path, tensors, metadata)


def _metadata_array(metadata: Mapping[str, str], name: str) -> list[Any]:
    raw = metadata.get(name)
    if not isinstance(raw, str):
        raise RawAnalysisError(f"safetensors metadata is missing JSON array {name}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RawAnalysisError(f"safetensors metadata {name} is not valid JSON") from exc
    if not isinstance(value, list):
        raise RawAnalysisError(f"safetensors metadata {name} must be a JSON array")
    return value


def _load_packed(path: Path, support: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, torch.Tensor], Mapping[str, list[Any]]]:
    try:
        with safe_open(path, framework="pt", device="cpu") as stream:
            names = set(stream.keys())
            header_metadata = stream.metadata() or {}
        tensors = load_file(path, device="cpu")
    except Exception as exc:
        raise RawAnalysisError(f"cannot load packed safetensors artifact {path}: {exc}") from exc
    if names != set(TENSOR_NAMES) or set(tensors) != set(TENSOR_NAMES):
        raise RawAnalysisError(f"packed artifact tensors must be exactly {sorted(TENSOR_NAMES)}")
    if set(header_metadata) != set(METADATA_NAMES):
        raise RawAnalysisError(f"packed artifact metadata keys must be exactly {sorted(METADATA_NAMES)}")
    metadata = {name: _metadata_array(header_metadata, name) for name in METADATA_NAMES}
    count = len(support)
    if any(len(values) != count for values in metadata.values()):
        raise RawAnalysisError("every packed metadata array must have one item per support row")

    expected_pairs = [row["pair_id"] for row in support]
    expected_stable = [row["stable_id"] for row in support]
    if metadata["pair_ids"] != expected_pairs:
        raise RawAnalysisError("packed pair_ids do not exactly match ordered completion support")
    if metadata["stable_ids"] != expected_stable:
        raise RawAnalysisError("packed stable_ids do not exactly match ordered completion support")
    integer_arrays = METADATA_NAMES - {"stable_ids"}
    for name in integer_arrays:
        minimum = 1 if name.endswith("_lengths") and "prompt" not in name else 0
        if any(type(value) is not int or value < minimum for value in metadata[name]):
            raise RawAnalysisError(f"packed metadata {name} contains an invalid integer")

    pos = tensors["positive_activations"]
    neg = tensors["negative_activations"]
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1:] != neg.shape[1:] or pos.shape[1] <= 0:
        raise RawAnalysisError("positive/negative activations must be packed rank-2 tensors with equal hidden size")
    if not pos.dtype.is_floating_point or pos.dtype != neg.dtype:
        raise RawAnalysisError("positive/negative activations must have the same floating dtype")
    if not bool(torch.isfinite(pos).all()) or not bool(torch.isfinite(neg).all()):
        raise RawAnalysisError("packed activations contain non-finite values")

    for polarity in ("positive", "negative"):
        lengths = metadata[f"{polarity}_lengths"]
        total = sum(lengths)
        activation = tensors[f"{polarity}_activations"]
        token_ids = tensors[f"{polarity}_token_ids"]
        mask = tensors[f"{polarity}_attention_mask"]
        if activation.shape[0] != total or token_ids.ndim != 1 or mask.ndim != 1:
            raise RawAnalysisError(f"{polarity} tensors do not implement packed ragged rank contract")
        if token_ids.shape[0] != total or mask.shape[0] != total:
            raise RawAnalysisError(f"{polarity} packed lengths do not equal tensor lengths")
        if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
            raise RawAnalysisError(f"{polarity}_token_ids must use an integral dtype")
        if mask.dtype.is_floating_point and mask.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            raise RawAnalysisError(f"{polarity}_attention_mask has invalid dtype")
        if not bool((mask == 1).all()):
            raise RawAnalysisError(f"{polarity}_attention_mask must be all-one valid packed tokens")
        prompts = metadata[f"{polarity}_prompt_lengths"]
        onsets = metadata[f"{polarity}_answer_onsets"]
        for index, (length, prompt, onset) in enumerate(zip(lengths, prompts, onsets)):
            if prompt > length:
                raise RawAnalysisError(f"{polarity} prompt length exceeds effective length at row {index}")
            if onset > length:
                raise RawAnalysisError(f"{polarity} answer onset exceeds effective length at row {index}")
            # Deliberately no equality/inference relation between prompt length and onset:
            # BPE boundaries and truncation make that relation extractor-specific evidence.
    return tensors, metadata


def _pair_slices(route: Route) -> Iterable[PairSlice]:
    positive_cursor = 0
    negative_cursor = 0
    for index, row in enumerate(route.support):
        pl = route.metadata["positive_lengths"][index]
        nl = route.metadata["negative_lengths"][index]
        yield PairSlice(
            row=row,
            positive=slice(positive_cursor, positive_cursor + pl),
            negative=slice(negative_cursor, negative_cursor + nl),
            positive_length=pl,
            negative_length=nl,
            positive_prompt_length=route.metadata["positive_prompt_lengths"][index],
            negative_prompt_length=route.metadata["negative_prompt_lengths"][index],
            positive_answer_onset=route.metadata["positive_answer_onsets"][index],
            negative_answer_onset=route.metadata["negative_answer_onsets"][index],
        )
        positive_cursor += pl
        negative_cursor += nl


def _fit_direction(route: Route) -> None:
    differences: list[torch.Tensor] = []
    pair_ids: list[int] = []
    stable_ids: list[str] = []
    pos_all = route.tensors["positive_activations"].to(torch.float64)
    neg_all = route.tensors["negative_activations"].to(torch.float64)
    for pair in _pair_slices(route):
        if pair.row["split"] != "train":
            continue
        positive = pos_all[pair.positive]
        negative = neg_all[pair.negative]
        count = min(
            pair.positive_length - pair.positive_answer_onset,
            pair.negative_length - pair.negative_answer_onset,
        )
        if count <= 0:
            continue
        differences.append(
            positive[pair.positive_answer_onset:pair.positive_answer_onset + count]
            - negative[pair.negative_answer_onset:pair.negative_answer_onset + count]
        )
        pair_ids.append(pair.row["pair_id"])
        stable_ids.append(pair.row["stable_id"])
    if not differences:
        raise RawAnalysisError("route has no aligned post-onset train tokens for direction fitting")
    direction = torch.cat(differences, dim=0).mean(dim=0)
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        raise RawAnalysisError("train-only mean difference direction is zero or non-finite")
    direction = direction / norm
    route.direction = direction
    route.direction_sha256 = hashlib.sha256(direction.numpy().tobytes(order="C")).hexdigest()
    route.direction_fit_pair_ids = tuple(pair_ids)
    route.direction_fit_stable_ids = tuple(stable_ids)


def _float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RawAnalysisError("analysis produced a non-finite scalar")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm = math.fsum(left) / len(left)
    rm = math.fsum(right) / len(right)
    centered_l = [value - lm for value in left]
    centered_r = [value - rm for value in right]
    denominator = math.sqrt(math.fsum(x * x for x in centered_l) * math.fsum(x * x for x in centered_r))
    if denominator == 0.0:
        return None
    return math.fsum(x * y for x, y in zip(centered_l, centered_r)) / denominator


def _analyze_route(route: Route) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert route.direction is not None and route.direction_sha256 is not None
    target = route.target
    pos_a = route.tensors["positive_activations"].to(torch.float64)
    neg_a = route.tensors["negative_activations"].to(torch.float64)
    pos_ids = route.tensors["positive_token_ids"]
    neg_ids = route.tensors["negative_token_ids"]
    rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for pair in _pair_slices(route):
        positive = pos_a[pair.positive]
        negative = neg_a[pair.negative]
        pids = pos_ids[pair.positive]
        nids = neg_ids[pair.negative]
        first_offset = max(-pair.positive_answer_onset, -pair.negative_answer_onset)
        stop_offset = min(
            pair.positive_length - pair.positive_answer_onset,
            pair.negative_length - pair.negative_answer_onset,
        )
        pair_rows: list[dict[str, Any]] = []
        for offset in range(first_offset, stop_offset):
            pi = pair.positive_answer_onset + offset
            ni = pair.negative_answer_onset + offset
            p = positive[pi]
            n = negative[ni]
            difference = p - n
            l2 = _float(torch.linalg.vector_norm(difference))
            denominator = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(n)
            cosine = 0.0 if float(denominator) == 0.0 else _float(torch.dot(p, n) / denominator)
            projection = _float(torch.dot(difference, route.direction))
            row = {
                "target_id": target["target_id"], "model": target["model"],
                "model_slug": target["model_slug"], "benchmark": target["benchmark"],
                "strategy": target["strategy"], "layer": target["layer"],
                "layer_count": target["layer_count"], "pair_id": pair.row["pair_id"],
                "stable_id": pair.row["stable_id"], "split": pair.row["split"],
                "offset": offset, "phase": "pre_onset" if offset < 0 else "post_onset",
                "positive_token_id": int(pids[pi]), "negative_token_id": int(nids[ni]),
                "positive_token_index": pi, "negative_token_index": ni,
                "l2_norm": l2, "cosine": cosine, "signed_projection": projection,
            }
            rows.append(row)
            pair_rows.append(row)
        if not pair_rows:
            raise RawAnalysisError(f"pair {pair.row['pair_id']} has no common onset-aligned tokens")
        norms = [row["l2_norm"] for row in pair_rows]
        maximum = max(norms)
        peak = next(row for row in pair_rows if row["l2_norm"] == maximum)
        pre = [row for row in pair_rows if row["offset"] < 0]
        post = [row for row in pair_rows if row["offset"] >= 0]
        pair_summaries.append({
            "pair_id": pair.row["pair_id"], "stable_id": pair.row["stable_id"],
            "split": pair.row["split"], "aligned_token_count": len(pair_rows),
            "first_offset": pair_rows[0]["offset"], "last_offset": pair_rows[-1]["offset"],
            "positive_prompt_length": pair.positive_prompt_length,
            "negative_prompt_length": pair.negative_prompt_length,
            "positive_answer_onset": pair.positive_answer_onset,
            "negative_answer_onset": pair.negative_answer_onset,
            "pre_onset_l2_auc": math.fsum(row["l2_norm"] for row in pre),
            "post_onset_l2_auc": math.fsum(row["l2_norm"] for row in post),
            "pre_onset_projection_auc": math.fsum(row["signed_projection"] for row in pre),
            "post_onset_projection_auc": math.fsum(row["signed_projection"] for row in post),
            "peak_l2_norm": maximum, "peak_offset": peak["offset"],
            "mean_l2_norm": _mean(norms),
            "mean_cosine": _mean([row["cosine"] for row in pair_rows]),
            "mean_signed_projection": _mean([row["signed_projection"] for row in pair_rows]),
        })
    split_counts = {split: sum(row["split"] == split for row in route.support)
                    for split in ("train", "validation", "test")}
    summary = {
        "target_id": target["target_id"], "model": target["model"],
        "model_slug": target["model_slug"], "benchmark": target["benchmark"],
        "strategy": target["strategy"], "layer": target["layer"],
        "layer_count": target["layer_count"],
        "normalized_depth": 0.0 if target["layer_count"] == 1 else
            (target["layer"] - 1) / (target["layer_count"] - 1),
        "pair_count": len(pair_summaries), "split_counts": split_counts,
        "token_count": len(rows),
        "mean_l2_norm": _mean([row["l2_norm"] for row in rows]),
        "mean_cosine": _mean([row["cosine"] for row in rows]),
        "mean_signed_projection": _mean([row["signed_projection"] for row in rows]),
        "mean_pre_onset_l2_auc": _mean([row["pre_onset_l2_auc"] for row in pair_summaries]),
        "mean_post_onset_l2_auc": _mean([row["post_onset_l2_auc"] for row in pair_summaries]),
        "mean_pre_onset_projection_auc": _mean([row["pre_onset_projection_auc"] for row in pair_summaries]),
        "mean_post_onset_projection_auc": _mean([row["post_onset_projection_auc"] for row in pair_summaries]),
        "direction": {
            "fit_split": "train", "fit_pair_ids": list(route.direction_fit_pair_ids),
            "fit_stable_ids": list(route.direction_fit_stable_ids),
            "test_pair_ids_read_for_fit": [], "unit_vector_sha256": route.direction_sha256,
        },
        "pairs": pair_summaries,
    }
    return rows, summary


def _layer_stability(token_rows: Sequence[Mapping[str, Any]], routes: Sequence[Route]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, str], dict[int, dict[int, Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in token_rows:
        key = (row["target_id"], row["strategy"], row["stable_id"], row["pair_id"], row["split"])
        if row["offset"] in grouped[key][row["layer"]]:
            raise RawAnalysisError("duplicate onset-relative token row")
        grouped[key][row["layer"]][row["offset"]] = row
    target_layers: dict[tuple[str, str], list[int]] = defaultdict(list)
    for route in routes:
        target_layers[(route.target["target_id"], route.target["strategy"])].append(route.target["layer"])
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        target_id, strategy, stable_id, pair_id, split = key
        layers = sorted(set(target_layers[(target_id, strategy)]))
        for lower, upper in zip(layers, layers[1:]):
            if upper != lower + 1 or lower not in grouped[key] or upper not in grouped[key]:
                continue
            common = sorted(set(grouped[key][lower]) & set(grouped[key][upper]))
            left = [grouped[key][lower][offset]["l2_norm"] for offset in common]
            right = [grouped[key][upper][offset]["l2_norm"] for offset in common]
            projections_left = [grouped[key][lower][offset]["signed_projection"] for offset in common]
            projections_right = [grouped[key][upper][offset]["signed_projection"] for offset in common]
            output.append({
                "target_id": target_id, "strategy": strategy, "pair_id": pair_id,
                "stable_id": stable_id, "split": split, "lower_layer": lower,
                "upper_layer": upper, "common_offsets": common,
                "common_token_count": len(common),
                "l2_pearson": _pearson(left, right),
                "signed_projection_pearson": _pearson(projections_left, projections_right),
            })
    return output


def _strategy_summaries(route_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in route_summaries:
        grouped[(row["target_id"], row["strategy"])].append(row)
    fields = (
        "mean_l2_norm", "mean_cosine", "mean_signed_projection",
        "mean_pre_onset_l2_auc", "mean_post_onset_l2_auc",
        "mean_pre_onset_projection_auc", "mean_post_onset_projection_auc",
    )
    output = []
    for (target_id, strategy), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: row["layer"])
        row: dict[str, Any] = {
            "target_id": target_id, "model": ordered[0]["model"],
            "model_slug": ordered[0]["model_slug"], "benchmark": ordered[0]["benchmark"],
            "strategy": strategy, "layers": [item["layer"] for item in ordered],
            "route_count": len(ordered), "pair_count_per_route": ordered[0]["pair_count"],
        }
        for field in fields:
            row[field] = _mean([item[field] for item in ordered if item[field] is not None])
        output.append(row)
    return output


def _depth_curves(route_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in route_summaries:
        grouped[(row["strategy"], row["normalized_depth"])].append(row)
    fields = ("mean_l2_norm", "mean_cosine", "mean_signed_projection",
              "mean_pre_onset_l2_auc", "mean_post_onset_l2_auc")
    output = []
    for (strategy, depth), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: (row["target_id"], row["layer"]))
        item: dict[str, Any] = {
            "strategy": strategy, "normalized_depth": depth,
            "target_count": len({row["target_id"] for row in ordered}),
            "route_count": len(ordered),
            "members": [{"target_id": row["target_id"], "layer": row["layer"]} for row in ordered],
        }
        for field in fields:
            item[field] = _mean([row[field] for row in ordered if row[field] is not None])
        output.append(item)
    return output


def _canonical_lines(rows: Iterable[Mapping[str, Any]]) -> bytes:
    data = b"".join(canonical_json(row) + b"\n" for row in rows)
    return data or b"\n"


def _write_new(path: Path, data: bytes) -> ContentRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RawAnalysisError(f"refusing to overwrite analysis artifact {path}") from exc
    digest = hashlib.sha256(data).hexdigest()
    ref = artifact_ref(path.name, digest, str(len(data)), digest)
    return ContentRef(ref["uri"], ref["generation"], len(data), ref["size"], ref["sha256"])


def _json_bytes(value: Any) -> bytes:
    return canonical_json(value) + b"\n"


def run(bundle_indexes: Sequence[Path], cache_dir: Path, output_dir: Path) -> Path:
    """Validate raw route completions and publish deterministic analyses create-only."""
    if not bundle_indexes:
        raise RawAnalysisError("at least one raw completion manifest is required")
    cache_dir = cache_dir.resolve()
    if not cache_dir.is_dir():
        raise RawAnalysisError(f"cache_dir does not exist: {cache_dir}")
    completion_paths = [Path(path).resolve() for path in bundle_indexes]
    if len(completion_paths) != len(set(completion_paths)):
        raise RawAnalysisError("completion manifest arguments contain duplicates")
    routes = [_validate_completion(path, cache_dir) for path in completion_paths]
    route_keys: set[tuple[str, str, int]] = set()
    target_manifest_refs: dict[str, Mapping[str, Any]] = {}
    target_identities: dict[str, tuple[Any, ...]] = {}
    for route in routes:
        key = (route.target["target_id"], route.target["strategy"], route.target["layer"])
        if key in route_keys:
            raise RawAnalysisError(f"duplicate target/strategy/layer route {key}")
        route_keys.add(key)
        target_id = route.target["target_id"]
        ref = route.completion["target_manifest_ref"]
        if target_id in target_manifest_refs and target_manifest_refs[target_id] != ref:
            raise RawAnalysisError("routes for one target bind different TargetManifest references")
        target_manifest_refs[target_id] = ref
        identity = (
            route.target["model"], route.target["model_slug"], route.target["benchmark"],
            route.target["layer_count"], tuple(sorted(route.completion["revisions"].items())),
            tuple((row["pair_id"], row["stable_id"], row["split"]) for row in route.support),
        )
        if target_id in target_identities and target_identities[target_id] != identity:
            raise RawAnalysisError("routes for one target disagree on revisions or ordered support identity")
        target_identities[target_id] = identity
    routes.sort(key=lambda route: (route.target["target_id"], route.target["strategy"], route.target["layer"]))
    for route in routes:
        _fit_direction(route)
    token_rows: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    for route in routes:
        rows, summary = _analyze_route(route)
        token_rows.extend(rows)
        route_summaries.append(summary)
    token_rows.sort(key=lambda row: (
        row["target_id"], row["strategy"], row["layer"], row["pair_id"], row["offset"],
    ))
    stability = _layer_stability(token_rows, routes)
    strategies = _strategy_summaries(route_summaries)
    curves = _depth_curves(route_summaries)

    destination = output_dir.resolve()
    if destination.exists():
        raise RawAnalysisError(f"analysis destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        refs: dict[str, ContentRef] = {}
        refs["token_trajectories"] = _write_new(staging / "token-trajectories.jsonl", _canonical_lines(token_rows))
        refs["route_summaries"] = _write_new(staging / "route-summaries.jsonl", _canonical_lines(route_summaries))
        refs["layer_stability"] = _write_new(staging / "layer-stability.jsonl", _canonical_lines(stability))
        refs["strategy_summaries"] = _write_new(staging / "strategy-summaries.jsonl", _canonical_lines(strategies))
        refs["cross_target_depth_curves"] = _write_new(staging / "cross-target-depth-curves.jsonl", _canonical_lines(curves))
        summary = {
            "schema_version": 1, "target_count": len(target_identities),
            "route_count": len(routes), "pair_route_count": sum(len(route.support) for route in routes),
            "token_row_count": len(token_rows), "strategy_summary_count": len(strategies),
            "layer_stability_count": len(stability), "depth_curve_count": len(curves),
            "splits_used_for_classifier_fit": ["train"], "test_rows_used_for_classifier_fit": 0,
        }
        refs["summary"] = _write_new(staging / "summary.json", _json_bytes(summary))
        provenance = {
            "schema_version": 1, "analysis": "desired_results_raw_analysis",
            "analysis_source_sha256": _digest_file(Path(__file__).resolve())[0],
            "completion_manifests": [{
                "target_id": route.target["target_id"], "strategy": route.target["strategy"],
                "layer": route.target["layer"], "path": route.completion_path.name,
                "manifest_sha256": route.completion_sha256,
                "target_manifest_ref": route.completion["target_manifest_ref"],
                "artifact_ref": route.artifact_ref.json(),
                "source_route_ref": route.completion["artifact"]["source_route_ref"],
                "revisions": route.completion["revisions"],
                "direction_fit_split": "train",
                "direction_fit_pair_ids": list(route.direction_fit_pair_ids),
                "direction_fit_stable_ids": list(route.direction_fit_stable_ids),
                "direction_sha256": route.direction_sha256,
            } for route in routes],
            "fit_policy": {
                "classifier": "unit_train_mean_positive_minus_negative_post_onset_v1",
                "fit_splits": ["train"], "selection_splits": [],
                "test_access_during_fit": False,
            },
            "alignment_policy": {
                "identity": "pair_id_and_stable_id_exact",
                "origin": "explicit_per_polarity_answer_onset",
                "answer_onset_inferred_from_token_length": False,
                "domain": "intersection_of_per_polarity_valid_attention_mask_tokens",
                "padding_read": False,
            },
        }
        refs["provenance"] = _write_new(staging / "provenance.json", _json_bytes(provenance))
        index_without_hash = {
            "schema_version": 1, "complete": True,
            "kind": "desired_results_raw_analysis",
            "artifacts": {name: ref.json() for name, ref in sorted(refs.items())},
            "summary": summary,
        }
        index = dict(index_without_hash)
        index["index_sha256"] = canonical_sha256(index_without_hash)
        _write_new(staging / "analysis-index.json", _json_bytes(index))
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "analysis-index.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_indexes", nargs="+", type=Path,
                        help="complete raw route manifest(s)")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args.bundle_indexes, args.cache_dir, args.output_dir)
    except (RawAnalysisError, OSError, ValueError) as exc:
        print(f"raw analysis failed: {exc}", file=os.sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
