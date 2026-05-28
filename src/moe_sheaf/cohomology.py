"""Sheaf cohomology computation for MoE routing manifolds.

Uses Vietoris-Rips filtration on the expert distance matrix to compute
persistent homology H⁰ and H¹. The sheaf structure modifies the
boundary operator via routing overlap between expert pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def _epsilon_default(dist_matrix: np.ndarray) -> float:
    """Pick a sensible default epsilon: the median pairwise distance."""
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    return float(np.median(upper))


def _h1_euler_approx(
    dist_matrix: np.ndarray,
    routing_overlap: np.ndarray | None,
    epsilon: float,
) -> float:
    """Fast H¹ estimate for large graphs using Euler characteristic.

    For a simplicial complex: χ = V - E + F
    H¹ = 1 - χ - (H⁰ - 1) = E - V - F + 1 - (H⁰ - 1)
    Approximate: H¹ ≈ E - V + 1 - H⁰ (ignoring triangles for speed, corrected)
    """
    n = dist_matrix.shape[0]
    adj = (dist_matrix <= epsilon)
    np.fill_diagonal(adj, False)
    n_edges = int(np.sum(adj) // 2)  # upper triangle count

    # Count triangles efficiently using matrix multiplication
    adj_f = adj.astype(np.float64)
    # (A³)ᵢᵢ counts paths of length 2 from i through i → triangles * 2
    a3 = adj_f @ adj_f @ adj_f
    n_triangles = int(np.trace(a3) / 6)  # each triangle counted 6 times

    # Euler characteristic: χ = V - E + F
    euler = n - n_edges + n_triangles

    # H⁰
    graph = csr_matrix(adj_f)
    h0, _ = connected_components(graph, directed=False)

    # H¹ ≈ 1 - χ - (h0 - 1) = E - V - F + 2 - h0
    h1 = n_edges - n - n_triangles + 2 - h0

    # Sheaf weighting: scale by routing overlap diversity
    if routing_overlap is not None:
        overlap_vals = routing_overlap[adj]
        if len(overlap_vals) > 0:
            diversity = 1.0 - float(np.mean(overlap_vals))
            h1 = h1 * (1.0 + diversity)

    return float(max(h1, 0.0))


def compute_h0(dist_matrix: np.ndarray, epsilon: float | None = None) -> int:
    """Compute H⁰ = number of connected components at filtration threshold epsilon.

    Args:
        dist_matrix: NxN pairwise distance matrix between experts.
        epsilon: Filtration threshold. If None, uses median distance.

    Returns:
        Number of connected components.
    """
    if epsilon is None:
        epsilon = _epsilon_default(dist_matrix)
    n = dist_matrix.shape[0]
    adj = (dist_matrix <= epsilon).astype(float)
    np.fill_diagonal(adj, 0)
    graph = csr_matrix(adj)
    n_components, _ = connected_components(graph, directed=False)
    return int(n_components)


def compute_h1(
    dist_matrix: np.ndarray,
    routing_overlap: np.ndarray | None = None,
    epsilon: float | None = None,
) -> float:
    """Compute H¹ of the routing sheaf at filtration threshold epsilon.

    For a Vietoris-Rips complex at scale epsilon:
    - H⁰ = #connected components
    - H¹ = dim(ker ∂₁) - dim(im ∂₂)  (simplified)

    We compute the rank deficiency of the 1-boundary operator, weighted
    by the sheaf's routing overlap, to capture sheaf-aware obstructions.

    When routing_overlap is None, this reduces to classical persistent H¹.

    Args:
        dist_matrix: NxN distance matrix.
        routing_overlap: NxN matrix of routing co-activation strengths.
        epsilon: Filtration threshold.

    Returns:
        H¹ dimension (float, as a weighted measure).
    """
    n = dist_matrix.shape[0]
    if n < 3:
        return 0.0
    if epsilon is None:
        epsilon = _epsilon_default(dist_matrix)

    # For large n, use Euler characteristic approximation
    if n > 50:
        return _h1_euler_approx(dist_matrix, routing_overlap, epsilon)

    # Build edge list for the Vietoris-Rips complex at scale epsilon
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= epsilon:
                edges.append((i, j))

    if len(edges) < 3:
        return 0.0

    # Build triangles (2-simplices)
    edge_set = set()
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in edges:
        edge_set.add((i, j))
        adj[i].add(j)
        adj[j].add(i)

    triangles = []
    seen_tri = set()
    for i in range(n):
        neighbors_i = adj[i]
        for j in neighbors_i:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j:
                    continue
                if k in neighbors_i:
                    tri = (i, j, k)
                    if tri not in seen_tri:
                        seen_tri.add(tri)
                        triangles.append(tri)

    n_edges = len(edges)
    edge_idx = {e: idx for idx, e in enumerate(edges)}

    # 1-boundary operator ∂₁: edges × vertices
    # ∂₁([i,j]) = eⱼ - eᵢ
    d1 = np.zeros((n, n_edges))
    for idx, (i, j) in enumerate(edges):
        d1[i, idx] = -1.0
        d1[j, idx] = 1.0

    # 2-boundary operator ∂₂: triangles × edges
    # ∂₂([i,j,k]) = [j,k] - [i,k] + [i,j]
    n_tri = len(triangles)
    d2 = np.zeros((n_edges, n_tri))
    for tri_idx, (i, j, k) in enumerate(triangles):
        for sign, a, b in [(1, i, j), (-1, i, k), (1, j, k)]:
            key = (min(a, b), max(a, b))
            if key in edge_idx:
                d2[edge_idx[key], tri_idx] += sign

    # Apply sheaf weighting from routing overlap
    if routing_overlap is not None:
        weights = np.ones(n_edges)
        for idx, (i, j) in enumerate(edges):
            w = float(routing_overlap[i, j])
            weights[idx] = max(w, 1e-8)
        # Weight the boundary operators
        W = np.diag(weights)
        d1 = d1 @ W
        d2 = W @ d2

    # H¹ = ker(∂₁) / im(∂₂)
    # dim(H¹) = dim(ker ∂₁) - dim(im ∂₂)
    # dim(ker ∂₁) = n_edges - rank(∂₁)
    # dim(im ∂₂) = rank(∂₂)
    rank_d1 = np.linalg.matrix_rank(d1, tol=1e-8)
    rank_d2 = np.linalg.matrix_rank(d2, tol=1e-8) if n_tri > 0 else 0
    dim_ker_d1 = n_edges - rank_d1
    h1 = dim_ker_d1 - rank_d2
    return float(max(h1, 0.0))


def persistence_diagram(
    dist_matrix: np.ndarray, max_dim: int = 1
) -> list[tuple[float, float]]:
    """Compute persistence diagram using a simple filtration.

    Sweeps epsilon from 0 to max distance, tracking births and deaths
    of H⁰ and H¹ features.

    Returns:
        List of (birth, death) pairs. death=inf for eternal features.
    """
    n = dist_matrix.shape[0]
    if n < 2:
        return []

    # Get sorted unique distances as filtration values
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    thresholds = np.sort(np.unique(upper))

    if len(thresholds) == 0:
        return []

    diagram: list[tuple[float, float]] = []

    # H⁰: track connected components
    prev_components = n  # each point is its own component
    component_births: dict[int, float] = {i: 0.0 for i in range(n)}

    for eps in thresholds:
        adj = (dist_matrix <= eps).astype(float)
        np.fill_diagonal(adj, 0)
        graph = csr_matrix(adj)
        n_comp, labels = connected_components(graph, directed=False)

        if n_comp < prev_components:
            # Some components merged — record deaths
            comp_ids: dict[int, list[int]] = {}
            for node in range(n):
                lbl = int(labels[node])
                comp_ids.setdefault(lbl, []).append(node)

            # The surviving component keeps the earliest birth;
            # the merged component dies here
            new_births: dict[int, float] = {}
            for lbl, nodes in comp_ids.items():
                births = [component_births.get(nd, 0.0) for nd in nodes]
                new_births[lbl] = min(births)
                # Record death for components that merged (born later)
                if len(nodes) > 1:
                    max_birth = max(births)
                    if max_birth > min(births):
                        diagram.append((max_birth, float(eps)))

            component_births = {nd: new_births[int(labels[nd])] for nd in range(n)}
            prev_components = n_comp

    # Eternal H⁰ features (never merge)
    seen_births = set()
    for b in component_births.values():
        if b not in seen_births:
            diagram.append((b, float("inf")))
            seen_births.add(b)

    # H¹ features: only compute for small n (performance)
    if max_dim >= 1 and n <= 30:
        prev_h1 = 0
        for idx, eps in enumerate(thresholds):
            h1 = compute_h1(dist_matrix, epsilon=eps)
            if h1 > prev_h1:
                # H¹ feature born
                if idx + 1 < len(thresholds):
                    diagram.append((float(eps), float(thresholds[idx + 1])))
                prev_h1 = h1

    return diagram
