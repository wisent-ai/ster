import os
import pkgutil

# Note: previous versions set os.environ["NUMBA_NUM_THREADS"] = "1" here,
# which caused RuntimeError when numba's reload_config (triggered by
# pynndescent import) detected the env and tried to re-launch threads at a
# different count than the one already running. Operators who actually want
# to constrain numba should set NUMBA_NUM_THREADS in the environment before
# launching python — not from inside our package init.

# pkgutil.extend_path lets sibling packages (wisent-extractors, wisent-evaluators)
# contribute to the wisent.* namespace from separate installed locations.
__path__ = pkgutil.extend_path(__path__, __name__)

# Legacy behavior: append direct child directories so existing code that relied
# on the custom __path__ layout keeps working.
_base = os.path.dirname(__file__)
for _entry in sorted(os.listdir(_base)):
    _path = os.path.join(_base, _entry)
    if os.path.isdir(_path) and not _entry.startswith((".", "_")):
        __path__.append(_path)

__version__ = "0.11.81"


def _wisent_install_hf_rate_limit_global() -> None:
    """Monkey-patch huggingface_hub.HfApi at top-level wisent import.

    Earlier the patch lived in wisent/core/reading/modules/utilities/data/
    sources/hf/__init__.py — but callers import that subpackage as
    `from wisent...hf.hf_writers import X`, which loads hf_writers.py
    directly without touching the subpackage's __init__.py. The
    monkey-patch never ran. Confirmed live on 2026-05-07: GCS token
    bucket showed 999/1000 tokens (untouched) while every job hit
    HF 429 errors. Installing here at top-level wisent guarantees the
    patch runs on first import of anything from the wisent package.
    """
    try:
        from huggingface_hub import HfApi
        from wisent_compute.providers.local.hf_rate import wait_for_hf_token
    except Exception:
        return
    if getattr(HfApi, "_wisent_rate_limit_installed", False):
        return
    for _m in ("upload_file", "upload_folder", "list_repo_tree",
               "preupload_lfs_files", "create_commit",
               # The download/info path: model_info, dataset_info, and
               # hf_hub_download (HfApi method) are what from_pretrained
               # calls under the hood. Not patching them here let every
               # agent's pip/transformers .from_pretrained() bypass the
               # fleet-wide GCS token bucket — confirmed by 429s in the
               # 'from_pretrained' call site of agent extraction logs
               # (c69fa5e3 et al, 2026-05-10).
               "model_info", "dataset_info", "repo_info",
               "list_repo_files", "hf_hub_download"):
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
    # Also wrap the MODULE-level huggingface_hub.hf_hub_download (and
    # snapshot_download) — transformers/datasets call those by import,
    # not by HfApi method, so HfApi-only patching misses them.
    try:
        import huggingface_hub as _hh
        for _fn in ("hf_hub_download", "snapshot_download"):
            _orig = getattr(_hh, _fn, None)
            if _orig is None or getattr(_orig, "_wisent_rate_limit_installed", False):
                continue
            def _make_mod(_o):
                def _w(*a, **k):
                    wait_for_hf_token()
                    return _o(*a, **k)
                _w._wisent_rate_limit_installed = True
                return _w
            setattr(_hh, _fn, _make_mod(_orig))
    except Exception:
        pass
_wisent_install_hf_rate_limit_global()

_LAZY_EXPORTS = {
    "OpenerPenaltyProcessor": (
        "wisent.core.control.tasks.base.diversity_processors",
        "OpenerPenaltyProcessor",
    ),
    "TriePenaltyProcessor": (
        "wisent.core.control.tasks.base.diversity_processors",
        "TriePenaltyProcessor",
    ),
    "PhraseLedger": (
        "wisent.core.control.tasks.base.diversity_processors",
        "PhraseLedger",
    ),
    "build_diversity_processors": (
        "wisent.core.control.tasks.base.diversity_processors",
        "build_diversity_processors",
    ),
    "Wisent": ("wisent.core.primitives.model_interface.core.wisent", "Wisent"),
    "TraitConfig": ("wisent.core.primitives.model_interface.core.wisent", "TraitConfig"),
    "Modality": ("wisent.core.primitives.models.modalities", "Modality"),
    "ModalityContent": (
        "wisent.core.primitives.models.modalities",
        "ModalityContent",
    ),
    "TextContent": ("wisent.core.primitives.models.modalities", "TextContent"),
    "AudioContent": ("wisent.core.primitives.models.modalities", "AudioContent"),
    "VideoContent": ("wisent.core.primitives.models.modalities", "VideoContent"),
    "ImageContent": ("wisent.core.primitives.models.modalities", "ImageContent"),
    "RobotState": ("wisent.core.primitives.models.modalities", "RobotState"),
    "RobotAction": ("wisent.core.primitives.models.modalities", "RobotAction"),
    "RobotTrajectory": (
        "wisent.core.primitives.models.modalities",
        "RobotTrajectory",
    ),
    "MultimodalContent": (
        "wisent.core.primitives.models.modalities",
        "MultimodalContent",
    ),
    "BaseAdapter": ("wisent.core.primitives.model_interface.adapters", "BaseAdapter"),
    "TextAdapter": ("wisent.core.primitives.model_interface.adapters", "TextAdapter"),
    "AudioAdapter": ("wisent.core.primitives.model_interface.adapters", "AudioAdapter"),
    "ImageAdapter": ("wisent.core.primitives.model_interface.adapters", "ImageAdapter"),
    "VideoAdapter": ("wisent.core.primitives.model_interface.adapters", "VideoAdapter"),
    "RoboticsAdapter": (
        "wisent.core.primitives.model_interface.adapters",
        "RoboticsAdapter",
    ),
    "MultimodalAdapter": (
        "wisent.core.primitives.model_interface.adapters",
        "MultimodalAdapter",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_EXPORTS[name]
    try:
        module = __import__(module_name, fromlist=[attr])
        value = getattr(module, attr)
    except ImportError:
        if name == "ImageAdapter":
            value = None
        else:
            raise
    globals()[name] = value
    return value

__all__ = [
    # Version
    "__version__",
    # Diversity processors
    "OpenerPenaltyProcessor",
    "TriePenaltyProcessor",
    "PhraseLedger",
    "build_diversity_processors",
    # Main interface
    "Wisent",
    "TraitConfig",
    # Modalities
    "Modality",
    "ModalityContent",
    "TextContent",
    "AudioContent",
    "VideoContent",
    "ImageContent",
    "RobotState",
    "RobotAction",
    "RobotTrajectory",
    "MultimodalContent",
    # Adapters
    "BaseAdapter",
    "TextAdapter",
    "AudioAdapter",
    "ImageAdapter",
    "VideoAdapter",
    "RoboticsAdapter",
    "MultimodalAdapter",
]
