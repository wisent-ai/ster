"""3D shape-aware renderers for Zwiad's geometry classification."""

from .dispatch import render_shape, compute_pca_3d
from .renderers import (
    render_linear_3d,
    render_cone_3d,
    render_orthogonal_3d,
    render_bimodal_3d,
    render_cluster_3d,
    render_manifold_3d,
    render_sparse_3d,
    render_sphere_3d,
)

__all__ = [
    "render_shape",
    "compute_pca_3d",
    "render_linear_3d",
    "render_cone_3d",
    "render_orthogonal_3d",
    "render_bimodal_3d",
    "render_cluster_3d",
    "render_manifold_3d",
    "render_sparse_3d",
    "render_sphere_3d",
]
