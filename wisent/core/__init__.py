import os as _os
import pkgutil as _pkgutil
# Merge with sibling installs of wisent.core.* (e.g. wisent-evaluators
# contributes wisent/core/reading/evaluators). Without this, the regular
# package above hides the namespace-package contributions in site-packages.
__path__ = _pkgutil.extend_path(__path__, __name__)
_base = _os.path.dirname(__file__)
for _entry in sorted(_os.listdir(_base)):
    _path = _os.path.join(_base, _entry)
    if _os.path.isdir(_path) and not _entry.startswith(('.', '_')):
        __path__.append(_path)
        for _sub in sorted(_os.listdir(_path)):
            _sub_path = _os.path.join(_path, _sub)
            if _os.path.isdir(_sub_path) and not _sub.startswith(('.', '_')):
                __path__.append(_sub_path)

_LAZY_EXPORTS = {
    "empty_device_cache": ("wisent.core.utils.infra_tools.infra", "empty_device_cache"),
    "preferred_dtype": ("wisent.core.utils.infra_tools.infra", "preferred_dtype"),
    "resolve_default_device": (
        "wisent.core.utils.infra_tools.infra",
        "resolve_default_device",
    ),
    "resolve_device": ("wisent.core.utils.infra_tools.infra", "resolve_device"),
    "resolve_torch_device": (
        "wisent.core.utils.infra_tools.infra",
        "resolve_torch_device",
    ),
    "analyze_steering_vector_subspace": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "analyze_steering_vector_subspace",
    ),
    "check_vector_quality": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "check_vector_quality",
    ),
    "compress_steering_vectors": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "compress_steering_vectors",
    ),
    "decompress_steering_vectors": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "decompress_steering_vectors",
    ),
    "compute_optimal_num_directions": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "compute_optimal_num_directions",
    ),
    "initialize_from_universal_basis": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "initialize_from_universal_basis",
    ),
    "verify_subspace_preservation": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "verify_subspace_preservation",
    ),
    "get_recommended_geometry_thresholds": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "get_recommended_geometry_thresholds",
    ),
    "SubspaceAnalysisConfig": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "SubspaceAnalysisConfig",
    ),
    "SubspaceAnalysisResult": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "SubspaceAnalysisResult",
    ),
    "UniversalBasis": (
        "wisent.core.control.steering_core.core.universal_subspace",
        "UniversalBasis",
    ),
    "compute_icd": ("wisent.core.reading.modules.runner.geometry_runner", "compute_icd"),
    "compute_nonsense_baseline": (
        "wisent.core.reading.modules.runner.geometry_runner",
        "compute_nonsense_baseline",
    ),
    "generate_nonsense_activations": (
        "wisent.core.reading.modules.runner.geometry_runner",
        "generate_nonsense_activations",
    ),
    "analyze_with_nonsense_baseline": (
        "wisent.core.reading.modules.runner.geometry_runner",
        "analyze_with_nonsense_baseline",
    ),
    "GeometryTestResult": (
        "wisent.core.reading.modules.runner.geometry_runner",
        "GeometryTestResult",
    ),
    "GeometrySearchResults": (
        "wisent.core.reading.modules.runner.geometry_runner",
        "GeometrySearchResults",
    ),
    "GeometryRunner": (
        "wisent.core.reading.modules.runner.geometry_runner",
        "GeometryRunner",
    ),
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
    # "SteeringMethod",
    # "SteeringType",
    "empty_device_cache",
    "preferred_dtype",
    "resolve_default_device",
    "resolve_device",
    "resolve_torch_device",
    # Universal Subspace Analysis
    "analyze_steering_vector_subspace",
    "check_vector_quality",
    "compress_steering_vectors",
    "decompress_steering_vectors",
    "compute_optimal_num_directions",
    "initialize_from_universal_basis",
    "verify_subspace_preservation",
    "get_recommended_geometry_thresholds",
    "SubspaceAnalysisConfig",
    "SubspaceAnalysisResult",
    "UniversalBasis",
    # Geometry analysis
    "compute_icd",
    "compute_nonsense_baseline",
    "generate_nonsense_activations",
    "analyze_with_nonsense_baseline",
    "GeometryTestResult",
    "GeometrySearchResults",
    "GeometryRunner",
]
