"""HF-Hub load helpers extracted from wisent_model.py.

The 1000-req/5min HF Hub ceiling is defended structurally by the
fleet-wide GCS token bucket at
wisent_compute.providers.local.hf_rate.wait_for_hf_token (installed
on every HfApi method that touches the network at top-level wisent
import). The 8-attempt 429 retry that used to wrap every HF call was
covering for gaps in that hook (download/info path was unhooked
until 0.11.35). Removed 2026-05-10. A 429 reaching this layer now
surfaces as a real error.
"""

from __future__ import annotations

from typing import Any, Callable


def _is_hf_rate_limit_exc(exc: BaseException) -> bool:
    cur: BaseException | None = exc
    while cur is not None:
        msg = str(cur)
        if (
            "429" in msg
            or "Too Many Requests" in msg
            or "rate limit" in msg.lower()
            or "hit the quota" in msg
            or "does not appear to have a file named" in msg
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _retry_on_hf_rate_limit(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    return fn(*args, **kwargs)


def _has_partial_safetensors_cache(model_name: str) -> bool:
    """True iff a local HF snapshot dir for model_name has a safetensors
    index listing shards that are missing on disk. Best-effort: any
    error returns False so the normal network path is unperturbed.

    transformers.from_pretrained, when it resolves a cached snapshot
    directory, opens shard files relative to that snapshot rather than
    re-fetching missing ones — so a partial cache (e.g. interrupted
    download that left model-00001-of-00002.safetensors but not
    model-00002) raises FileNotFoundError mid-load instead of healing
    itself. Confirmed live 2026-05-19: job 9c6592e3 (gpt-oss-20b/hrm8k
    on local@ubuntu-server) failed with
    `FileNotFoundError: ...models--openai--gpt-oss-20b/snapshots/.../
    model-00002-of-00002.safetensors` after the local_files_only=True
    pre-attempt also failed.
    """
    try:
        import json
        from pathlib import Path
        from huggingface_hub import constants
        hub = Path(constants.HF_HUB_CACHE)
        safe = "models--" + model_name.replace("/", "--")
        snap = hub / safe / "snapshots"
        if not snap.is_dir():
            return False
        for d in snap.iterdir():
            idx = d / "model.safetensors.index.json"
            if not idx.is_file():
                continue
            with idx.open("r") as f:
                doc = json.load(f)
            for shard in set((doc.get("weight_map") or {}).values()):
                if not (d / shard).is_file():
                    return True
        return False
    except Exception:
        return False


def _load_cache_first(
    fn: Callable[..., Any], model_name: str, **kwargs: Any
) -> Any:
    """Local cache first; on miss, ONE network attempt. If the local
    cache snapshot is partial (index lists shards not on disk) force a
    clean re-download on the network call so transformers does not
    FileNotFound mid-load against the broken snapshot."""
    if kwargs.get("local_files_only"):
        return fn(model_name, **kwargs)
    try:
        return fn(model_name, local_files_only=True, **kwargs)
    except Exception:
        pass
    if "force_download" not in kwargs and _has_partial_safetensors_cache(model_name):
        kwargs = {**kwargs, "force_download": True}
    return fn(model_name, **kwargs)


__all__ = [
    "_is_hf_rate_limit_exc",
    "_retry_on_hf_rate_limit",
    "_load_cache_first",
    "_has_partial_safetensors_cache",
]
