"""Layer DAG recovery for Zwiad.

Builds a directed acyclic graph over transformer layers capturing which
layers carry concept signal and which layers' signal is mediated by earlier
layers' signal. The minimum source set of the recovered DAG is the smallest
causal set of layers needed to steer the concept --- "the layers we are
steering on."

Algorithm: PC-style skeleton + orientation specialized to the layer index
ordering. The transformer's structural prior fixes the direction (only
earlier layers can be parents of later layers), so the search reduces to
edge pruning via conditional-independence tests on per-layer scalar
features (each sample's activation projected onto that layer's mean-
difference direction). Conditional independence is tested by partial
correlation; a separating set of size up to ``max_conditioning_set`` is
searched between every pair of layers ordered (i, j) with i < j.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


@dataclass
class LayerDAGResult:
    """Result of layer DAG recovery for a concept."""
    nodes: List[int]
    edges: List[Tuple[int, int]]
    adjacency: Dict[int, List[int]]
    minimum_steering_set: List[int]
    alpha: float
    max_conditioning_set: int
    signal_layers: List[int]


def extract_layer_features(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[int, np.ndarray]:
    """Per layer, project each sample onto the mean-difference direction.

    Returns a 1-D scalar per (sample, layer): the inner product of the
    sample activation with the unit mean-difference direction
    ``v_layer = (mu_pos - mu_neg) / ||mu_pos - mu_neg||``. Output array per
    layer has shape ``(n_pos + n_neg,)``.
    """
    if not activations_by_layer:
        raise ValueError("activations_by_layer is empty; nothing to extract")
    features: Dict[int, np.ndarray] = {}
    for layer, payload in activations_by_layer.items():
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise ValueError(
                f"layer {layer}: activations must be a (pos, neg) tuple"
            )
        pos, neg = payload
        if not isinstance(pos, torch.Tensor) or not isinstance(neg, torch.Tensor):
            raise TypeError(f"layer {layer}: pos and neg must be torch.Tensor")
        if pos.ndim != 2 or neg.ndim != 2:
            raise ValueError(
                f"layer {layer}: expected 2-D activations; "
                f"got {tuple(pos.shape)}, {tuple(neg.shape)}"
            )
        pos_np = pos.detach().cpu().float().numpy()
        neg_np = neg.detach().cpu().float().numpy()
        mu_pos = pos_np.mean(axis=0)
        mu_neg = neg_np.mean(axis=0)
        v = mu_pos - mu_neg
        norm = float(np.linalg.norm(v))
        if norm == 0:
            raise ValueError(
                f"layer {layer}: mean-difference vector has zero norm; cannot project"
            )
        v_hat = v / norm
        proj_pos = pos_np @ v_hat
        proj_neg = neg_np @ v_hat
        features[layer] = np.concatenate([proj_pos, proj_neg], axis=0)
    lengths = {len(arr) for arr in features.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"layer feature arrays have inconsistent lengths: {sorted(lengths)}"
        )
    return features


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[float, float]:
    """Partial correlation of ``x`` and ``y`` given ``z``.

    ``z`` may have shape ``(n,)`` or ``(n, k)``. When ``z`` is empty
    (k == 0), reduces to the unconditioned Pearson correlation. Returns
    ``(rho, p_value)`` using the Fisher z-transform with degrees of freedom
    ``n - 2 - k``.
    """
    from scipy import stats
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = int(len(x))
    if n != len(y):
        raise ValueError(f"x and y must have equal length; got {n} vs {len(y)}")
    if z is None:
        Z = np.zeros((n, 0))
    elif hasattr(z, "ndim") and z.ndim == 1:
        Z = z.reshape(-1, 1)
    else:
        Z = np.asarray(z)
        if Z.ndim != 2:
            raise ValueError(f"z must be 1-D or 2-D; got ndim={Z.ndim}")
        if Z.shape[0] != n:
            raise ValueError(f"z first dim must match n={n}; got {Z.shape[0]}")
    k = int(Z.shape[1])
    if k == 0:
        rho, p = stats.pearsonr(x, y)
        return float(rho), float(p)
    Z_full = np.column_stack([np.ones(n), Z])
    coef_x, *_ = np.linalg.lstsq(Z_full, x, rcond=None)
    coef_y, *_ = np.linalg.lstsq(Z_full, y, rcond=None)
    res_x = x - Z_full @ coef_x
    res_y = y - Z_full @ coef_y
    rho, _ = stats.pearsonr(res_x, res_y)
    if abs(rho) >= 1.0:
        return float(rho), 0.0
    df = n - 2 - k
    if df <= 1:
        raise ValueError(
            f"degrees of freedom must be > 1 for Fisher z; got n={n}, k={k}, df={df}"
        )
    fisher_z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(df - 1)
    z_stat = fisher_z / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    return float(rho), float(p)


def recover_layer_dag(
    layer_features: Dict[int, np.ndarray],
    alpha: float,
    max_conditioning_set: int,
) -> Dict[str, Any]:
    """Recover a DAG over layers using PC-style CI tests with layer-order prior.

    Layers are inherently ordered by index. Candidate edges are restricted to
    (i, j) with i < j. For each candidate, an edge survives only if there is
    no subset of in-between layers (up to size ``max_conditioning_set``) that
    renders i and j conditionally independent at significance ``alpha``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if max_conditioning_set < 0:
        raise ValueError(
            f"max_conditioning_set must be non-negative; got {max_conditioning_set}"
        )
    if not layer_features:
        raise ValueError("layer_features is empty; cannot recover a DAG")
    layers = sorted(layer_features.keys())
    if len(layers) < 2:
        return {"nodes": list(layers), "edges": [], "adjacency": {l: [] for l in layers}}
    n_samples = int(len(layer_features[layers[0]]))
    edges_remaining: List[Tuple[int, int]] = []
    for i_idx, i in enumerate(layers):
        for j in layers[i_idx + 1:]:
            edges_remaining.append((i, j))

    edges_removed: set = set()
    for k_cond in range(0, max_conditioning_set + 1):
        for (i, j) in list(edges_remaining):
            if (i, j) in edges_removed:
                continue
            between = [l for l in layers if i < l < j]
            if len(between) < k_cond:
                continue
            xi = layer_features[i]
            xj = layer_features[j]
            separated = False
            for combo in combinations(between, k_cond):
                if len(combo) == 0:
                    rho, p = partial_correlation(xi, xj, np.zeros((n_samples, 0)))
                else:
                    Z = np.column_stack([layer_features[l] for l in combo])
                    rho, p = partial_correlation(xi, xj, Z)
                if p > alpha:
                    separated = True
                    break
            if separated:
                edges_removed.add((i, j))

    edges_final = [e for e in edges_remaining if e not in edges_removed]
    adjacency: Dict[int, List[int]] = {l: [] for l in layers}
    for (i, j) in edges_final:
        adjacency[i].append(j)
    return {"nodes": list(layers), "edges": edges_final, "adjacency": adjacency}


def minimum_steering_set(
    adjacency: Dict[int, List[int]],
    signal_layers: List[int],
) -> List[int]:
    """Minimum set of layers whose downstream closure covers all signal layers.

    Greedy set cover (NP-hard exact; greedy is the standard O(log n) bound).
    For each candidate layer, computes the downstream-reachable layer set;
    iteratively picks the layer that covers the most uncovered signal layers,
    marks them covered, and repeats. Returns the chosen set, sorted.
    """
    if not signal_layers:
        return []
    layers = sorted(adjacency.keys())
    if not layers:
        raise ValueError("adjacency is empty; cannot compute minimum steering set")
    signal_set = set(signal_layers)
    unknown = signal_set - set(layers)
    if unknown:
        raise ValueError(
            f"signal_layers contains entries absent from adjacency: {sorted(unknown)}"
        )

    def downstream(node: int) -> set:
        visited: set = set()
        stack = [node]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            for child in adjacency.get(x, []):
                if child not in visited:
                    stack.append(child)
        return visited

    closures: Dict[int, set] = {l: downstream(l) for l in layers}
    uncovered = set(signal_set)
    chosen: List[int] = []
    while uncovered:
        best_layer = None
        best_cover = -1
        for l in layers:
            cov = len(closures[l] & uncovered)
            if cov > best_cover:
                best_cover = cov
                best_layer = l
        if best_layer is None or best_cover == 0:
            chosen.extend(sorted(uncovered))
            uncovered.clear()
            break
        chosen.append(best_layer)
        uncovered -= closures[best_layer]
    return sorted(set(chosen))


def test_layer_dag(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    signal_layers: List[int],
    alpha: float,
    max_conditioning_set: int,
) -> LayerDAGResult:
    """Step: recover layer DAG + compute minimum steering set."""
    features = extract_layer_features(activations_by_layer)
    dag = recover_layer_dag(features, alpha, max_conditioning_set)
    min_set = minimum_steering_set(dag["adjacency"], signal_layers)
    return LayerDAGResult(
        nodes=dag["nodes"],
        edges=dag["edges"],
        adjacency=dag["adjacency"],
        minimum_steering_set=min_set,
        alpha=float(alpha),
        max_conditioning_set=int(max_conditioning_set),
        signal_layers=list(signal_layers),
    )


def result_to_dict(result: LayerDAGResult) -> Dict[str, Any]:
    """JSON-serializable dict for protocol output."""
    return {
        "nodes": list(result.nodes),
        "edges": [list(e) for e in result.edges],
        "adjacency": {str(k): list(v) for k, v in result.adjacency.items()},
        "minimum_steering_set": list(result.minimum_steering_set),
        "alpha": result.alpha,
        "max_conditioning_set": result.max_conditioning_set,
        "signal_layers": list(result.signal_layers),
    }
