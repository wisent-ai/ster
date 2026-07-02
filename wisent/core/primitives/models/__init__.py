import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

_LAZY_EXPORTS = {
    "WisentModel": ("wisent.core.primitives.models.wisent_model", "WisentModel"),
    "InferenceConfig": ("wisent.core.primitives.models.config", "InferenceConfig"),
    "get_config": ("wisent.core.primitives.models.config", "get_config"),
    "set_config": ("wisent.core.primitives.models.config", "set_config"),
    "save_config": ("wisent.core.primitives.models.config", "save_config"),
    "update_config": ("wisent.core.primitives.models.config", "update_config"),
    "reset_config": ("wisent.core.primitives.models.config", "reset_config"),
    "get_generate_kwargs": (
        "wisent.core.primitives.models.config",
        "get_generate_kwargs",
    ),
    "CONFIG_FILE": ("wisent.core.primitives.models.config", "CONFIG_FILE"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_EXPORTS[name]
    module = __import__(module_name, fromlist=[attr])
    value = getattr(module, attr)
    globals()[name] = value
    return value

__all__ = [
    "WisentModel",
    "InferenceConfig",
    "get_config",
    "set_config",
    "save_config",
    "update_config",
    "reset_config",
    "get_generate_kwargs",
    "CONFIG_FILE",
]
