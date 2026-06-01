"""Unsupervised topology discovery via Vietoris-Rips persistent homology.

For a contrastive activation cloud, computes Betti numbers
(beta_0, beta_1, beta_2) from the persistence diagram of a Vietoris-Rips
filtration, matches against a named-shape lookup, and returns the raw
Betti signature when no match exists. Discovers topology without any shape
prior, complementing the supervised closed-vocabulary detectors in
``geometry_detectors.py``.

Every parameter is required and explicit; missing or invalid arguments
raise ``ValueError``. ``ripser`` is required.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class TopologyTestResult:
    """Result of unsupervised topology discovery on a contrastive cloud."""
    betti_pos: Tuple[int, ...]
    betti_neg: Tuple[int, ...]
    betti_union: Tuple[int, ...]
    named_shape_pos: Optional[str]
    named_shape_neg: Optional[str]
    named_shape_union: Optional[str]
    persistence_entropy_pos: Dict[int, float]
    persistence_entropy_neg: Dict[int, float]
    persistence_entropy_union: Dict[int, float]
    max_persistence_pos: Dict[int, float]
    max_persistence_neg: Dict[int, float]
    max_persistence_union: Dict[int, float]
    max_dim: int
    max_edge_length: float
    persistence_threshold: float


# Named-shape lookup keyed by Betti signature. Conservative: only entries
# whose Betti tuple identifies a shape modulo common ambiguities (Klein bottle
# vs circle over Z/2 are not distinguishable at this level and are excluded).
# Returns None for unknown signatures so the caller can still inspect the raw
# Betti tuple.
_NAMED_SHAPES: Dict[Tuple[int, ...], str] = {
    (1, 0, 0): "point",
    (1, 1, 0): "circle",
    (1, 0, 1): "sphere",
    (1, 2, 1): "torus",
    (1, 2, 0): "figure_eight",
    (1, 3, 0): "wedge_three_circles",
    (1, 4, 0): "wedge_four_circles",
    (1, 0, 2): "wedge_two_spheres",
    (2, 0, 0): "two_blobs",
    (2, 2, 0): "two_circles",
    (2, 1, 0): "one_blob_one_circle",
    (3, 0, 0): "three_blobs",
    (3, 3, 0): "three_circles",
    (4, 0, 0): "four_blobs",
    (1, 2, 2): "double_torus",
    (1, 4, 1): "two_tori_disjoint",
}


def compute_persistent_homology(
    cloud: torch.Tensor,
    max_dim: int,
    max_edge_length: float,
) -> Dict[int, np.ndarray]:
    """Vietoris-Rips persistent homology of a point cloud, via ``ripser``."""
    if max_dim < 0:
        raise ValueError(f"max_dim must be non-negative; got {max_dim}")
    if max_edge_length <= 0:
        raise ValueError(f"max_edge_length must be positive; got {max_edge_length}")
    if cloud.ndim != 2:
        raise ValueError(
            f"cloud must be 2-D (n_points, n_features); got shape {tuple(cloud.shape)}"
        )
    n_points = int(cloud.shape[0])
    if n_points < 2:
        raise ValueError(
            f"cloud needs at least 2 points for Vietoris-Rips persistence; got {n_points}"
        )
    try:
        from ripser import ripser
    except ImportError as e:
        raise ImportError(
            "Persistent-homology topology discovery requires 'ripser'. "
            "Install with: pip install ripser"
        ) from e

    X = cloud.detach().cpu().float().numpy()
    result = ripser(X, maxdim=int(max_dim), thresh=float(max_edge_length))
    diagrams = result["dgms"]
    return {d: np.asarray(diagrams[d]) for d in range(max_dim + 1)}


def compute_betti_signature(
    persistence_diagrams: Dict[int, np.ndarray],
    persistence_threshold: float,
) -> Tuple[int, ...]:
    """Count features per dimension whose persistence exceeds the threshold.

    Infinite-persistence features (death = inf) are always counted.
    """
    if persistence_threshold < 0:
        raise ValueError(
            f"persistence_threshold must be non-negative; got {persistence_threshold}"
        )
    if not persistence_diagrams:
        raise ValueError("persistence_diagrams is empty; nothing to count")
    max_dim = max(persistence_diagrams.keys())
    betti = []
    for d in range(max_dim + 1):
        if d not in persistence_diagrams:
            raise ValueError(f"persistence_diagrams missing dimension {d}")
        dgm = persistence_diagrams[d]
        if len(dgm) == 0:
            betti.append(0)
            continue
        deaths = dgm[:, 1]
        births = dgm[:, 0]
        infinite = np.isinf(deaths)
        finite_persistent = (~infinite) & ((deaths - births) > persistence_threshold)
        betti.append(int(infinite.sum() + finite_persistent.sum()))
    return tuple(betti)


def compute_persistence_entropy(
    persistence_diagrams: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """Shannon entropy of normalized finite persistences per dimension."""
    entropies: Dict[int, float] = {}
    for d, dgm in persistence_diagrams.items():
        if len(dgm) == 0:
            entropies[d] = 0.0
            continue
        finite_mask = ~np.isinf(dgm[:, 1])
        if not finite_mask.any():
            entropies[d] = 0.0
            continue
        pers = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        total = float(pers.sum())
        if total <= 0:
            entropies[d] = 0.0
            continue
        p = pers / total
        entropies[d] = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
    return entropies


def compute_max_persistence(
    persistence_diagrams: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """Longest finite persistence bar per dimension. Returns 0.0 when none exist."""
    out: Dict[int, float] = {}
    for d, dgm in persistence_diagrams.items():
        if len(dgm) == 0:
            out[d] = 0.0
            continue
        finite_mask = ~np.isinf(dgm[:, 1])
        if not finite_mask.any():
            out[d] = 0.0
            continue
        pers = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        out[d] = float(pers.max())
    return out


def identify_named_shape(betti: Tuple[int, ...]) -> Optional[str]:
    """Look up a named topology for a Betti signature; None if not in table."""
    if not isinstance(betti, tuple):
        raise TypeError(f"betti must be a tuple; got {type(betti)}")
    return _NAMED_SHAPES.get(betti)


def test_topology(
    pos: torch.Tensor,
    neg: torch.Tensor,
    max_dim: int,
    max_edge_length: float,
    persistence_threshold: float,
) -> TopologyTestResult:
    """Step: unsupervised topology of pos cloud, neg cloud, and pos union neg."""
    if not isinstance(pos, torch.Tensor) or not isinstance(neg, torch.Tensor):
        raise TypeError(
            f"pos and neg must be torch.Tensor; got {type(pos)}, {type(neg)}"
        )
    if pos.ndim != 2 or neg.ndim != 2:
        raise ValueError(
            f"pos and neg must be 2-D (n_points, n_features); "
            f"got {tuple(pos.shape)}, {tuple(neg.shape)}"
        )
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"pos and neg must share feature dim; got {pos.shape[1]} vs {neg.shape[1]}"
        )

    union = torch.cat([pos, neg], dim=0)

    dgm_pos = compute_persistent_homology(pos, max_dim, max_edge_length)
    dgm_neg = compute_persistent_homology(neg, max_dim, max_edge_length)
    dgm_union = compute_persistent_homology(union, max_dim, max_edge_length)

    betti_pos = compute_betti_signature(dgm_pos, persistence_threshold)
    betti_neg = compute_betti_signature(dgm_neg, persistence_threshold)
    betti_union = compute_betti_signature(dgm_union, persistence_threshold)

    return TopologyTestResult(
        betti_pos=betti_pos,
        betti_neg=betti_neg,
        betti_union=betti_union,
        named_shape_pos=identify_named_shape(betti_pos),
        named_shape_neg=identify_named_shape(betti_neg),
        named_shape_union=identify_named_shape(betti_union),
        persistence_entropy_pos=compute_persistence_entropy(dgm_pos),
        persistence_entropy_neg=compute_persistence_entropy(dgm_neg),
        persistence_entropy_union=compute_persistence_entropy(dgm_union),
        max_persistence_pos=compute_max_persistence(dgm_pos),
        max_persistence_neg=compute_max_persistence(dgm_neg),
        max_persistence_union=compute_max_persistence(dgm_union),
        max_dim=int(max_dim),
        max_edge_length=float(max_edge_length),
        persistence_threshold=float(persistence_threshold),
    )


def result_to_dict(result: TopologyTestResult) -> Dict[str, Any]:
    """JSON-serializable dict for the protocol output."""
    return {
        "betti_pos": list(result.betti_pos),
        "betti_neg": list(result.betti_neg),
        "betti_union": list(result.betti_union),
        "named_shape_pos": result.named_shape_pos,
        "named_shape_neg": result.named_shape_neg,
        "named_shape_union": result.named_shape_union,
        "persistence_entropy_pos": {str(k): v for k, v in result.persistence_entropy_pos.items()},
        "persistence_entropy_neg": {str(k): v for k, v in result.persistence_entropy_neg.items()},
        "persistence_entropy_union": {str(k): v for k, v in result.persistence_entropy_union.items()},
        "max_persistence_pos": {str(k): v for k, v in result.max_persistence_pos.items()},
        "max_persistence_neg": {str(k): v for k, v in result.max_persistence_neg.items()},
        "max_persistence_union": {str(k): v for k, v in result.max_persistence_union.items()},
        "max_dim": result.max_dim,
        "max_edge_length": result.max_edge_length,
        "persistence_threshold": result.persistence_threshold,
    }
