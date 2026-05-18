"""
3D shape-renderer dispatcher.

render_shape(shape, pos, neg, ...) reads the SHAPE_MAP string from
Zwiad's GeometryProfile.activation_shape and routes to the matching
renderer in renderers.py. Unknown shapes return a plain 3D PCA scatter
without overlay — no silent substitution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from wisent.core.utils.config_tools.constants import (
    DEFAULT_RANDOM_SEED,
    VIZ_N_COMPONENTS_3D,
)


__all__ = ["render_shape", "compute_pca_3d"]


def compute_pca_3d(
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint PCA to 3D so pos/neg live in the same projected basis.

    Returns:
        pos_3d, neg_3d, explained_variance_ratio.
    """
    from sklearn.decomposition import PCA

    pos_np = pos.float().cpu().numpy()
    neg_np = neg.float().cpu().numpy()
    combined = np.concatenate([pos_np, neg_np], axis=0)
    n_target = min(VIZ_N_COMPONENTS_3D, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=n_target, random_state=DEFAULT_RANDOM_SEED)
    proj = pca.fit_transform(combined)
    if proj.shape[1] < VIZ_N_COMPONENTS_3D:
        pad = np.zeros((proj.shape[0], VIZ_N_COMPONENTS_3D - proj.shape[1]))
        proj = np.concatenate([proj, pad], axis=1)
    pos_3d = proj[: pos_np.shape[0]]
    neg_3d = proj[pos_np.shape[0] :]
    return pos_3d, neg_3d, np.asarray(pca.explained_variance_ratio_)


def render_shape(
    shape: str,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    cluster_labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dispatch on Zwiad's SHAPE_MAP string.

    Args:
        shape: one of linear/cone/orthogonal/bimodal/cluster/manifold/sparse/sphere
        pos:   [N, D] positive activations.
        neg:   [N, D] negative activations.
        cluster_labels: optional pre-computed cluster labels for shape='cluster'.
        title: optional figure title.

    Returns:
        Dict containing pos_3d, neg_3d, shape-specific overlay geometry,
        explained_variance_ratio, title. Unknown shape returns shape='raw'
        plus the bare 3D scatter without overlay.
    """
    from . import renderers as _r

    _DISPATCH = {
        "linear": _r.render_linear_3d,
        "cone": _r.render_cone_3d,
        "orthogonal": _r.render_orthogonal_3d,
        "bimodal": _r.render_bimodal_3d,
        "cluster": _r.render_cluster_3d,
        "manifold": _r.render_manifold_3d,
        "sparse": _r.render_sparse_3d,
        "sphere": _r.render_sphere_3d,
    }
    fn = _DISPATCH.get(shape)
    if fn is None:
        pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
        return {
            "shape": shape or "raw",
            "pos_3d": pos_3d,
            "neg_3d": neg_3d,
            "explained_variance_ratio": evr,
            "title": title or f"{shape or 'raw'} (3D)",
        }
    if shape == "cluster":
        return fn(pos, neg, cluster_labels=cluster_labels, title=title or "Cluster (3D)")
    if title is None:
        return fn(pos, neg)
    return fn(pos, neg, title=title)
