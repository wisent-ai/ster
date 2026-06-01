"""Unsupervised geometry-discovery extensions for Zwiad.

Re-exports the two new modules' public surfaces under a single namespace.
"""
from .topology import (
    TopologyTestResult,
    compute_persistent_homology,
    compute_betti_signature,
    compute_persistence_entropy,
    compute_max_persistence,
    identify_named_shape,
    test_topology,
    result_to_dict as topology_result_to_dict,
)
from .layer_dag import (
    LayerDAGResult,
    extract_layer_features,
    partial_correlation,
    recover_layer_dag,
    minimum_steering_set,
    test_layer_dag,
    result_to_dict as layer_dag_result_to_dict,
)

__all__ = [
    "TopologyTestResult",
    "compute_persistent_homology",
    "compute_betti_signature",
    "compute_persistence_entropy",
    "compute_max_persistence",
    "identify_named_shape",
    "test_topology",
    "topology_result_to_dict",
    "LayerDAGResult",
    "extract_layer_features",
    "partial_correlation",
    "recover_layer_dag",
    "minimum_steering_set",
    "test_layer_dag",
    "layer_dag_result_to_dict",
]
