"""HuggingFace Hub integration for activation storage."""

# Monkey-patch HfApi's request-issuing methods at module-import time so
# every HF API call across the wisent fleet (regardless of who created the
# HfApi instance) acquires a token from a GCS-backed shared rate-limit
# bucket first. HF accounts are limited to 1000 req/5min account-wide;
# with 6+ parallel agents this ceiling is hit constantly without
# coordination, producing 429s that fail random jobs. The bucket lives in
# wisent_compute.providers.local.hf_rate; refill at 200 tokens/min matches
# the published HF limit. No-op if wisent_compute isn't installed.
def _wisent_install_hf_rate_limit() -> None:
    try:
        from huggingface_hub import HfApi
        from wisent_compute.providers.local.hf_rate import wait_for_hf_token
    except Exception:
        return
    if getattr(HfApi, "_wisent_rate_limit_installed", False):
        return
    for _m in ("upload_file", "upload_folder", "list_repo_tree",
               "preupload_lfs_files", "create_commit"):
        _orig = getattr(HfApi, _m, None)
        if _orig is None:
            continue
        def _make(_o):
            def _w(self, *a, **k):
                wait_for_hf_token()
                return _o(self, *a, **k)
            return _w
        setattr(HfApi, _m, _make(_orig))
    HfApi._wisent_rate_limit_installed = True
_wisent_install_hf_rate_limit()

from .hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    activation_hf_path,
    model_to_safe_name,
    pair_texts_hf_path,
    raw_activation_hf_path,
    safe_name_to_model,
)
from .hf_loaders import (
    load_activations_from_hf,
    load_available_layers_from_hf,
    load_pair_texts_from_hf,
)
from .hf_writers import (
    write_marker,
    consolidate_index,
    upload_activation_shard,
    upload_pair_texts,
    upload_raw_activation_shard,
    flush_staging_dir,
)
from .migration import (
    migrate_activation_table,
    migrate_pair_texts,
    migrate_raw_activation_table,
)
from .migration_verify import migrate_all, verify_migration

__all__ = [
    "HF_REPO_ID",
    "HF_REPO_TYPE",
    "activation_hf_path",
    "load_activations_from_hf",
    "load_available_layers_from_hf",
    "load_pair_texts_from_hf",
    "model_to_safe_name",
    "pair_texts_hf_path",
    "raw_activation_hf_path",
    "safe_name_to_model",
    "write_marker",
    "consolidate_index",
    "upload_activation_shard",
    "upload_pair_texts",
    "upload_raw_activation_shard",
    "flush_staging_dir",
    "migrate_activation_table",
    "migrate_all",
    "migrate_pair_texts",
    "migrate_raw_activation_table",
    "verify_migration",
]
