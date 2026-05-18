"""
Per-shape 3D renderers for Zwiad's geometry classification.

Each function returns a dict with pos_3d, neg_3d, the shape-specific
overlay geometry, explained_variance_ratio, and title — matching the
contract of the existing 2D plot helpers in _viz_basic.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from wisent.core.utils.config_tools.constants import (
    DEFAULT_RANDOM_SEED,
    NORM_EPS,
)
from .dispatch import compute_pca_3d


__all__ = [
    "render_linear_3d",
    "render_cone_3d",
    "render_orthogonal_3d",
    "render_bimodal_3d",
    "render_cluster_3d",
    "render_manifold_3d",
    "render_sparse_3d",
    "render_sphere_3d",
]


def render_linear_3d(pos, neg, title: str = "Linear (3D)") -> Dict[str, Any]:
    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    direction = pos_3d.mean(axis=0) - neg_3d.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm > NORM_EPS:
        direction = direction / norm
    span = float(np.linalg.norm(np.concatenate([pos_3d, neg_3d], axis=0).std(axis=0)))
    midpoint = 0.5 * (pos_3d.mean(axis=0) + neg_3d.mean(axis=0))
    axis_endpoints = np.stack([midpoint - span * direction, midpoint + span * direction])
    return {
        "shape": "linear",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "axis_endpoints": axis_endpoints,
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_cone_3d(pos, neg, title: str = "Cone (3D)") -> Dict[str, Any]:
    activations = torch.cat([pos, neg], dim=0)
    X = activations.float().cpu().numpy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid = norms.squeeze() > NORM_EPS
    X_norm = X[valid] / norms[valid]
    mean_dir_hd = X_norm.mean(axis=0)
    mean_dir_hd = mean_dir_hd / (np.linalg.norm(mean_dir_hd) + NORM_EPS)
    cos_angles = X_norm @ mean_dir_hd
    angles = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
    mean_angle = float(angles.mean())

    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    combined_3d = np.concatenate([pos_3d, neg_3d], axis=0)
    apex = combined_3d.mean(axis=0)
    centered = combined_3d - apex
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis_3d = vt[0]
    half_angle = np.radians(mean_angle)
    height = float(np.abs(centered @ axis_3d).max())
    radius = height * float(np.tan(half_angle))
    thetas = np.linspace(0.0, 2.0 * np.pi, 48)
    perp = vt[1] if vt.shape[0] > 1 else np.array([0.0, 1.0, 0.0])
    perp2 = vt[2] if vt.shape[0] > 2 else np.cross(axis_3d, perp)
    ring = apex + height * axis_3d + radius * (
        np.outer(np.cos(thetas), perp) + np.outer(np.sin(thetas), perp2)
    )
    cone_lines = np.stack([np.broadcast_to(apex, ring.shape), ring], axis=1)
    return {
        "shape": "cone",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "cone_apex": apex,
        "cone_axis": axis_3d,
        "cone_half_angle_deg": mean_angle,
        "cone_lines": cone_lines,
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_orthogonal_3d(pos, neg, title: str = "Orthogonal (3D)") -> Dict[str, Any]:
    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    combined = np.concatenate([pos_3d, neg_3d], axis=0)
    origin = combined.mean(axis=0)
    span = float(combined.std(axis=0).max())
    pc1 = np.array([span, 0.0, 0.0])
    pc2 = np.array([0.0, span, 0.0])
    axes = np.stack([origin - pc1, origin + pc1, origin - pc2, origin + pc2]).reshape(2, 2, 3)
    return {
        "shape": "orthogonal",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "axes": axes,
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_bimodal_3d(pos, neg, title: str = "Bimodal (3D)") -> Dict[str, Any]:
    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    c_pos = pos_3d.mean(axis=0)
    c_neg = neg_3d.mean(axis=0)
    normal = c_pos - c_neg
    n_norm = float(np.linalg.norm(normal))
    normal_unit = normal / (n_norm + NORM_EPS)
    midpoint = 0.5 * (c_pos + c_neg)
    return {
        "shape": "bimodal",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "centroid_pos": c_pos,
        "centroid_neg": c_neg,
        "separating_plane_point": midpoint,
        "separating_plane_normal": normal_unit,
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_cluster_3d(pos, neg, cluster_labels: Optional[np.ndarray] = None, title: str = "Cluster (3D)") -> Dict[str, Any]:
    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    diff_3d = pos_3d - neg_3d
    if cluster_labels is None:
        from sklearn.cluster import KMeans
        n_clusters = min(8, max(2, diff_3d.shape[0] // 4))
        km = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_SEED, n_init=10)
        labels_arr = km.fit_predict(diff_3d)
    else:
        labels_arr = np.asarray(cluster_labels)
    unique = np.unique(labels_arr)
    centroids = np.stack([diff_3d[labels_arr == c].mean(axis=0) for c in unique])
    return {
        "shape": "cluster",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "diff_3d": diff_3d,
        "cluster_labels": labels_arr,
        "cluster_centroids": centroids,
        "n_clusters": int(unique.size),
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_manifold_3d(pos, neg, n_neighbors: int = 5, title: str = "Manifold (3D)") -> Dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    diff_3d = pos_3d - neg_3d
    k = min(n_neighbors + 1, diff_3d.shape[0])
    nn = NearestNeighbors(n_neighbors=k).fit(diff_3d)
    _, idx = nn.kneighbors(diff_3d)
    edges = []
    for i in range(diff_3d.shape[0]):
        for j in idx[i, 1:]:
            edges.append(np.stack([diff_3d[i], diff_3d[j]]))
    edges_arr = np.asarray(edges) if edges else np.zeros((0, 2, 3))
    return {
        "shape": "manifold",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "diff_3d": diff_3d,
        "edges": edges_arr,
        "n_neighbors": n_neighbors,
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_sparse_3d(pos, neg, title: str = "Sparse (3D)") -> Dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    combined = np.concatenate([pos_3d, neg_3d], axis=0)
    k = min(5 + 1, combined.shape[0])
    nn = NearestNeighbors(n_neighbors=k).fit(combined)
    dists, _ = nn.kneighbors(combined)
    local_density = 1.0 / (dists[:, 1:].mean(axis=1) + NORM_EPS)
    density_norm = (local_density - local_density.min()) / (
        local_density.max() - local_density.min() + NORM_EPS
    )
    return {
        "shape": "sparse",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "density": density_norm,
        "explained_variance_ratio": evr,
        "title": title,
    }


def render_sphere_3d(pos, neg, title: str = "Sphere (3D)") -> Dict[str, Any]:
    pos_3d, neg_3d, evr = compute_pca_3d(pos, neg)
    combined = np.concatenate([pos_3d, neg_3d], axis=0)
    center = combined.mean(axis=0)
    radii = np.linalg.norm(combined - center, axis=1)
    radius = float(radii.mean())
    radius_std = float(radii.std())
    u_grid = np.linspace(0.0, 2.0 * np.pi, 32)
    v_grid = np.linspace(0.0, np.pi, 16)
    u, v = np.meshgrid(u_grid, v_grid)
    sphere_x = center[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = center[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = center[2] + radius * np.cos(v)
    return {
        "shape": "sphere",
        "pos_3d": pos_3d,
        "neg_3d": neg_3d,
        "sphere_center": center,
        "sphere_radius": radius,
        "sphere_radius_std": radius_std,
        "sphere_surface": np.stack([sphere_x, sphere_y, sphere_z], axis=-1),
        "explained_variance_ratio": evr,
        "title": title,
    }
