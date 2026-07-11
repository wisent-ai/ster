#!/usr/bin/env python3
"""Build local completion proofs from pinned HF safetensors headers only."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BENCHMARK = "winogrande"
REPO_ID = "wisent-ai/activations"
REVISION = "8c01dd5342f5b13c6d62eca9c343cd9714ec2e9b"
STRATEGIES = (
    "chat_first", "chat_last", "chat_mean", "chat_max_norm", "chat_weighted",
    "mc_balanced", "role_play",
)
LAYERS = tuple(range(1, 17))
EXPECTED_PAIRS = 500
MAX_HEADER_BYTES = 16 * 1024 * 1024
DEFAULT_INVENTORY = (
    Path(__file__).resolve().parents[3]
    / ".work/results_scope/desired_results_state_v1/result_inventory.sqlite"
)


class PreflightError(RuntimeError):
    """Remote activation state does not satisfy the frozen target contract."""


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read JSON {path}: {exc}") from exc


def _compact_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _headers() -> Dict[str, str]:
    headers = {"User-Agent": "wisent-desired-results-preflight/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _url(repo_path: str) -> str:
    quoted = urllib.parse.quote(repo_path, safe="/")
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{quoted}"


def _range(repo_path: str, start: int, end: int) -> bytes:
    headers = _headers()
    headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(_url(repo_path), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise PreflightError(
                    f"HF did not honor byte range for {repo_path}: HTTP {response.status}"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise PreflightError(
                    f"HF returned unexpected Content-Range for {repo_path}: {content_range!r}"
                )
            expected = end - start + 1
            data = response.read(expected + 1)
            if len(data) != expected:
                raise PreflightError(
                    f"HF returned {len(data)} bytes for {repo_path}; expected {expected}"
                )
            return data
    except urllib.error.HTTPError as exc:
        raise PreflightError(f"HF range request failed for {repo_path}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise PreflightError(f"HF range request failed for {repo_path}: {exc.reason}") from exc


def _download_json(repo_path: str) -> Tuple[Any, str]:
    request = urllib.request.Request(_url(repo_path), headers=_headers())
    digest = hashlib.sha256()
    chunks = []
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise PreflightError(f"HF JSON request failed for {repo_path}: {exc}") from exc
    try:
        return json.loads(b"".join(chunks)), digest.hexdigest()
    except json.JSONDecodeError as exc:
        raise PreflightError(f"HF JSON artifact is malformed: {repo_path}") from exc


def _manifest(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    unit = data.get("job_unit", {}) if isinstance(data, dict) else {}
    scope = data.get("activation_search_scope", {}) if isinstance(data, dict) else {}
    revisions = data.get("revisions", {}) if isinstance(data, dict) else {}
    if unit.get("model") != MODEL or unit.get("benchmark") != BENCHMARK:
        raise PreflightError("manifest is not the pinned model x benchmark target")
    if revisions.get("activation") != REVISION:
        raise PreflightError("manifest activation revision is not the pinned immutable revision")
    if tuple(scope.get("extraction_strategies", ())) != STRATEGIES:
        raise PreflightError("manifest does not contain the frozen seven strategies")
    if tuple(scope.get("layers", ())) != LAYERS:
        raise PreflightError("manifest does not contain the frozen 16 layers")
    if scope.get("extraction_component") != "residual_stream":
        raise PreflightError("manifest extraction component is not residual_stream")
    split_ids = data.get("split", {}).get("pair_ids", {})
    if set(split_ids) != {"train", "validation", "test"}:
        raise PreflightError("manifest split support is incomplete")
    support = sorted(value for values in split_ids.values() for value in values)
    if support != list(range(EXPECTED_PAIRS)):
        raise PreflightError("manifest support is not the exact ordered 0..499 target")
    return data


def _pair_text_check(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    artifact, raw_sha256 = _download_json("pair_texts/winogrande.json")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("pairs"), list):
        raise PreflightError("pair-text artifact has no pairs list")
    pairs = artifact["pairs"]
    if len(pairs) < EXPECTED_PAIRS:
        raise PreflightError("pair-text artifact does not cover target support")
    selected = pairs[:EXPECTED_PAIRS]
    for pair_id, pair in enumerate(selected):
        if not isinstance(pair, dict) or not isinstance(pair.get("prompt"), str):
            raise PreflightError(f"invalid pair text at pair_id {pair_id}")
        positive = pair.get("positive_response")
        negative = pair.get("negative_response")
        if not isinstance(positive, dict) or not isinstance(positive.get("model_response"), str):
            raise PreflightError(f"invalid positive pair text at pair_id {pair_id}")
        if not isinstance(negative, dict) or not isinstance(negative.get("model_response"), str):
            raise PreflightError(f"invalid negative pair text at pair_id {pair_id}")
    pair_text_hash = _compact_hash(selected)
    expected = manifest.get("input_identity", {}).get("pair_text_hash")
    if pair_text_hash != expected:
        raise PreflightError(
            f"pair-text hash mismatch: expected {expected}, got {pair_text_hash}"
        )
    return {
        "hf_path": "pair_texts/winogrande.json",
        "raw_sha256": raw_sha256,
        "pair_text_hash": pair_text_hash,
        "pair_ids": list(range(EXPECTED_PAIRS)),
        "artifact_pair_count": len(pairs),
    }


def _inventory_support(inventory: Path, manifest: Mapping[str, Any]) -> Sequence[str]:
    if not inventory.is_file():
        raise PreflightError(f"inventory does not exist: {inventory}")
    connection = sqlite3.connect(f"file:{inventory.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        targets = connection.execute(
            "SELECT target_id, activation_revision, pair_count, layer_count, "
            "pair_text_hash, support_hash FROM prepared_targets "
            "WHERE model_name=? AND benchmark=?", (MODEL, BENCHMARK),
        ).fetchall()
        if len(targets) != 1:
            raise PreflightError("inventory does not contain exactly one pinned target")
        target = targets[0]
        rows = connection.execute(
            "SELECT pair_id, stable_id FROM prepared_target_support "
            "WHERE target_id=? ORDER BY pair_id", (target["target_id"],),
        ).fetchall()
    finally:
        connection.close()
    if (
        target["activation_revision"] != REVISION
        or target["pair_count"] != EXPECTED_PAIRS
        or target["layer_count"] != len(LAYERS)
        or target["pair_text_hash"] != manifest["input_identity"]["pair_text_hash"]
        or target["support_hash"] != manifest["input_identity"]["support_hash"]
    ):
        raise PreflightError("inventory target identity does not match the preflight manifest")
    if [row["pair_id"] for row in rows] != list(range(EXPECTED_PAIRS)):
        raise PreflightError("inventory support is not exact ordered 0..499")
    stable_ids = [row["stable_id"] for row in rows]
    if any(not isinstance(value, str) or not value for value in stable_ids):
        raise PreflightError("inventory contains an invalid stable_id")
    return stable_ids


def _activation_path(strategy: str, layer: int) -> str:
    return (
        "activations/meta-llama__Llama-3.2-1B-Instruct/winogrande/"
        f"{strategy}/layer_{layer}.safetensors"
    )


def _route(strategy: str, layer: int) -> Dict[str, Any]:
    repo_path = _activation_path(strategy, layer)
    prefix = _range(repo_path, 0, 7)
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        raise PreflightError(
            f"invalid safetensors header length {header_length} for {repo_path}"
        )
    raw_header = _range(repo_path, 8, 7 + header_length)
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid safetensors header JSON for {repo_path}") from exc
    if not isinstance(header, dict):
        raise PreflightError(f"safetensors header is not an object for {repo_path}")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("pair_ids"), str):
        raise PreflightError(f"safetensors metadata lacks pair_ids for {repo_path}")
    try:
        pair_ids = json.loads(metadata["pair_ids"])
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid pair_ids metadata for {repo_path}") from exc
    if pair_ids != list(range(EXPECTED_PAIRS)):
        raise PreflightError(f"activation support is not exact ordered 0..499 for {repo_path}")
    tensor_shapes = {}
    tensor_dtypes = {}
    for name in ("pos_activations", "neg_activations"):
        tensor = header.get(name)
        if not isinstance(tensor, dict):
            raise PreflightError(f"missing {name} tensor header for {repo_path}")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(shape, list) or len(shape) != 2
            or shape[0] != EXPECTED_PAIRS or type(shape[1]) is not int or shape[1] <= 0
            or not isinstance(dtype, str)
            or not isinstance(offsets, list) or len(offsets) != 2
        ):
            raise PreflightError(f"invalid {name} tensor header for {repo_path}")
        tensor_shapes[name] = shape
        tensor_dtypes[name] = dtype
    if tensor_shapes["pos_activations"] != tensor_shapes["neg_activations"]:
        raise PreflightError(f"positive/negative tensor shapes differ for {repo_path}")
    if tensor_dtypes["pos_activations"] != tensor_dtypes["neg_activations"]:
        raise PreflightError(f"positive/negative tensor dtypes differ for {repo_path}")
    return {
        "schema_version": 1,
        "complete": True,
        "proof_source": "pinned_hf_safetensors_header",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "hf_path": repo_path,
        "model": MODEL,
        "benchmark": BENCHMARK,
        "extraction_component": "residual_stream",
        "extraction_strategy": strategy,
        "layers": [layer],
        "pair_ids": pair_ids,
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "safetensors_header_length": header_length,
        "safetensors_header_sha256": hashlib.sha256(raw_header).hexdigest(),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run(manifest_path: Path, inventory: Path, output: Path, workers: int) -> Path:
    manifest_path = manifest_path.resolve()
    manifest = _manifest(manifest_path)
    expected_stable_ids = _inventory_support(inventory.resolve(), manifest)
    output = output.resolve()
    if output.exists():
        raise PreflightError(f"destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        pair_text = _pair_text_check(manifest)
        requested = [(strategy, layer) for strategy in STRATEGIES for layer in LAYERS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_route, strategy, layer): (strategy, layer)
                for strategy, layer in requested
            }
            proofs = {}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                proofs[key] = future.result()
        if set(proofs) != set(requested):
            raise PreflightError("not every strategy/layer route was proven")
        proof_dir = staging / "proofs"
        proof_dir.mkdir()
        artifacts = []
        for strategy, layer in requested:
            proof_name = f"{strategy}.layer_{layer}.json"
            _atomic_json(proof_dir / proof_name, proofs[(strategy, layer)])
            artifacts.append({
                "extraction_strategy": strategy,
                "layer": layer,
                "completion_manifest": f"proofs/{proof_name}",
            })
        index = {
            "schema_version": 1,
            "complete": True,
            "proof_source": "pinned_hf_safetensors_headers",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "model": MODEL,
            "benchmark": BENCHMARK,
            "extraction_component": "residual_stream",
            "route_count": len(artifacts),
            "pair_count": EXPECTED_PAIRS,
            "pair_ids": list(range(EXPECTED_PAIRS)),
            "pair_text": pair_text,
            "pair_text_hash": manifest["input_identity"]["pair_text_hash"],
            "support_hash": manifest["input_identity"]["support_hash"],
            "stable_ids_sha256": _compact_hash(expected_stable_ids),
            "preflight_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "artifacts": artifacts,
            "submission_performed": False,
            "model_loaded": False,
            "tensor_payload_downloaded": False,
        }
        _atomic_json(staging / "completion-index.json", index)
        os.replace(staging, output)
        return output / "completion-index.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.workers < 1 or args.workers > 32:
            raise PreflightError("--workers must be between 1 and 32")
        print(run(args.manifest, args.inventory, args.output, args.workers))
        return 0
    except (PreflightError, OSError) as exc:
        print(f"activation preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
