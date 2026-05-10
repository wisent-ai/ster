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


def _load_cache_first(
    fn: Callable[..., Any], model_name: str, **kwargs: Any
) -> Any:
    """Local cache first; on miss, ONE network attempt."""
    if kwargs.get("local_files_only"):
        return fn(model_name, **kwargs)
    try:
        return fn(model_name, local_files_only=True, **kwargs)
    except Exception:
        pass
    return fn(model_name, **kwargs)


__all__ = [
    "_is_hf_rate_limit_exc",
    "_retry_on_hf_rate_limit",
    "_load_cache_first",
]
