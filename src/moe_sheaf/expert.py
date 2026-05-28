"""Expert manifold representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Expert:
    """A single MoE expert represented as a point on a weight manifold.

    Attributes:
        id: Unique expert identifier.
        weight_matrix: The expert's projection/feed-forward weight matrix (input_dim, output_dim).
        activation_stats: Dictionary with keys 'mean', 'std', 'sparsity' describing
            the distribution of activations routed to this expert.
    """

    id: int
    weight_matrix: np.ndarray
    activation_stats: dict[str, float] = field(default_factory=lambda: {"mean": 0.0, "std": 1.0, "sparsity": 0.0})

    def __post_init__(self):
        self.weight_matrix = np.asarray(self.weight_matrix, dtype=np.float64)

    @property
    def input_dim(self) -> int:
        return self.weight_matrix.shape[0]

    @property
    def output_dim(self) -> int:
        return self.weight_matrix.shape[1]

    @property
    def num_params(self) -> int:
        return self.weight_matrix.size

    def manifold_point(self) -> np.ndarray:
        """Flatten weight matrix to a point on the expert manifold."""
        return self.weight_matrix.flatten()

    def distance_to(self, other: Expert) -> float:
        """Euclidean distance between experts on the weight manifold."""
        return float(np.linalg.norm(self.manifold_point() - other.manifold_point()))

    def cosine_similarity(self, other: Expert) -> float:
        """Cosine similarity between expert weight vectors."""
        a, b = self.manifold_point(), other.manifold_point()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @classmethod
    def random(cls, id: int, input_dim: int = 64, output_dim: int = 32, seed: int | None = None) -> Expert:
        """Create a random expert for testing."""
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((input_dim, output_dim))
        return cls(id=id, weight_matrix=w, activation_stats={"mean": 0.0, "std": 1.0, "sparsity": 0.5})
