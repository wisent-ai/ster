import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""CLI execution logic for Wisent commands.

All imports are lazy to avoid circular import chains on Python 3.10
when this package is loaded as a side effect of namespace resolution.
"""

_IMPORT_MAP = {
    'execute_tasks': '.analysis.config.tasks',
    'execute_inference_config': '.analysis.config.inference_config_cli',
    'execute_diagnose_pairs': '.analysis.diagnosis.diagnose_pairs',
    'execute_diagnose_vectors': '.analysis.diagnosis.diagnose_vectors',
    'execute_cluster_benchmarks': '.analysis.diagnosis.cluster_benchmarks',
    'execute_evaluate_responses': '.analysis.evaluation.evaluate_responses',
    'execute_evaluate_refusal': '.analysis.evaluation.evaluate_refusal',
    'execute_check_linearity': '.analysis.evaluation.check_linearity',
    'execute_get_activations': '.analysis.geometry.get_activations',
    'execute_geometry_search': '.analysis.geometry.geometry_search',
    'execute_zwiad': '.analysis.geometry.zwiad',
    'execute_modify_weights': '.analysis.training.modify_weights',
    'execute_train_unified_goodness': '.analysis.training.train_unified_goodness',
    'execute_generate_pairs_from_task': '.generation.pairs.generate_pairs_from_task',
    'execute_generate_pairs': '.generation.pairs.generate_pairs',
    'execute_generate_vector_from_task': '.generation.vectors.generate_vector_from_task',
    'execute_generate_vector_from_synthetic': '.generation.vectors.generate_vector_from_synthetic',
    'execute_generate_responses': '.generation.vectors.generate_responses',
    'execute_optimize_steering': '.optimization.core.optimize_steering',
    'execute_optimization_cache': '.optimization.core.optimization_cache',
    'execute_optimize': '.optimization.core.optimize',
    'execute_optimize_classification': '.optimization.specific.optimize_classification',
    'execute_optimize_sample_size': '.optimization.specific.optimize_sample_size',
    'execute_optimize_weights': '.optimization.specific.optimize_weights',
    'execute_create_steering_object': '.steering.core.create_steering_object',
    'execute_multi_steer': '.steering.core.multi_steer',
    'execute_verify_steering': '.steering.core.verify_steering',
    'execute_discover_steering': '.steering.core.discover_steering',
    'execute_steering_viz': '.steering.viz.steering_viz',
    'execute_per_concept_steering_viz': '.steering.viz.per_concept_steering_viz',
    'execute_agent': '.agent.main',
    'execute_migrate_activations': '.data.migrate_activations',
    'execute_find_best_method': '.optimize_steering.pipeline.find_best.handler',
    'execute_axbench': '.concepts.axbench_cmd',
}


def __getattr__(name):
    if name == 'execute_compare_steering':
        from wisent.core.reading.modules.utilities.comparison import execute_compare_steering
        return execute_compare_steering
    if name in _IMPORT_MAP:
        import importlib
        mod = importlib.import_module(_IMPORT_MAP[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_IMPORT_MAP.keys()) + ['execute_compare_steering']
