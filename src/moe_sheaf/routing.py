"""MoE routing interpreted as a sheaf on the expert manifold."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from moe_sheaf.expert import Expert
from moe_sheaf.cohomology import compute_h0, compute_h1, persistence_diagram


class MoESheaf:
    """Model an MoE layer as a sheaf on the expert routing manifold.

    The experts form a discrete space (points on a weight manifold).
    The *stalk* at each expert is its weight vector and activation statistics.
    *Restriction maps* are defined by the routing overlap between neighboring experts.
    The topology is determined by a Vietoris-Rips filtration on pairwise distances.

    Args:
        experts: List of N experts.
        routing_weights: Array of shape (num_tokens, N) giving routing probabilities.
    """

    def __init__(self, experts: list[Expert], routing_weights: np.ndarray):
        self.experts = experts
        self.routing_weights = np.asarray(routing_weights, dtype=np.float64)
        if self.routing_weights.ndim == 1:
            self.routing_weights = self.routing_weights.reshape(1, -1)
        self.n_experts = len(experts)
        assert self.routing_weights.shape[1] == self.n_experts

        # Precompute distance matrix
        points = np.stack([e.manifold_point() for e in self.experts])
        self._dist_matrix = squareform(pdist(points, metric="euclidean"))

        # Routing overlap matrix: how often expert pairs are co-activated
        rw = self.routing_weights
        top_k = (rw > np.sort(rw, axis=1)[:, [-2]]).astype(float) if rw.shape[1] > 2 else (rw > 0.1).astype(float)
        self._routing_overlap = top_k.T @ top_k
        # Normalize to [0, 1]
        max_val = self._routing_overlap.max()
        if max_val > 0:
            self._routing_overlap /= max_val

    def stalk(self, expert_id: int) -> dict:
        """The stalk at each expert: its weight state and activation statistics."""
        e = self.experts[expert_id]
        return {
            "id": e.id,
            "weight_norm": float(np.linalg.norm(e.manifold_point())),
            "manifold_point": e.manifold_point(),
            "activation_stats": dict(e.activation_stats),
            "routing_load": float(self.routing_weights[:, expert_id].mean()),
        }

    def restriction_map(self, expert_i: int, expert_j: int) -> np.ndarray:
        """Restriction map between neighboring experts.

        Defined as the element-wise product of normalized weight vectors,
        scaled by routing overlap. For non-neighbors (beyond epsilon) this
        is the zero map.
        """
        pi = self.experts[expert_i].manifold_point()
        pj = self.experts[expert_j].manifold_point()

        ni = np.linalg.norm(pi)
        nj = np.linalg.norm(pj)
        if ni == 0 or nj == 0:
            return np.zeros_like(pi)

        overlap = self._routing_overlap[expert_i, expert_j]
        return overlap * (pi / ni) * (pj / nj)

    def routing_manifold(self) -> np.ndarray:
        """Return the distance matrix of the routing manifold."""
        return self._dist_matrix.copy()

    def compute_h0(self, epsilon: float | None = None) -> int:
        """Compute H⁰ (number of connected components) of the routing sheaf."""
        return compute_h0(self._dist_matrix, epsilon)

    def compute_h1(self, epsilon: float | None = None) -> float:
        """Compute H¹ (obstruction) of the routing sheaf at given filtration threshold."""
        return compute_h1(self._dist_matrix, self._routing_overlap, epsilon)

    def persistence_diagram(self) -> list[tuple[float, float]]:
        """Persistence diagram of the expert routing topology."""
        return persistence_diagram(self._dist_matrix)

    @property
    def distance_matrix(self) -> np.ndarray:
        return self._dist_matrix.copy()
