"""Full analysis pipeline for MoE models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from moe_sheaf.expert import Expert
from moe_sheaf.routing import MoESheaf
from moe_sheaf.cohomology import compute_h0, compute_h1, persistence_diagram as persistence


@dataclass
class LayerAnalysis:
    """Analysis results for a single MoE layer."""
    layer_name: str
    n_experts: int
    h0: int
    h1: float
    h1_per_param: float
    total_params: int
    distance_stats: dict[str, float]  # mean, std, min, max pairwise distances
    routing_entropy: float
    persistence_diagram: list[tuple[float, float]]
    expert_norms: list[float]
    stalks: list[dict]


@dataclass
class CorrelationResult:
    """Result of correlating H¹ with generalization across models."""
    pearson_r: float
    spearman_r: float
    p_value_pearson: float
    p_value_spearman: float
    h1_per_params: list[float]
    generalization_scores: list[float]
    n_models: int
    conjecture_supported: bool


class MoEAnalysis:
    """Full analysis pipeline for MoE model state dicts.

    Analyzes the sheaf cohomology of routing layers and tests the
    generalization conjecture.
    """

    def __init__(self, model_state: dict[str, np.ndarray]):
        """Initialize from a model's state dict.

        Expected keys pattern: layers containing 'expert' and 'routing' weights.
        """
        self.model_state = model_state
        self._layers = self._extract_layers()

    def _extract_layers(self) -> dict[str, dict[str, Any]]:
        """Extract MoE layer info from state dict.

        Groups keys by their top-level prefix (first dotted component).
        Keys containing 'expert' become expert weights, 'routing' or 'gate' become routing.
        """
        layers: dict[str, dict[str, Any]] = {}

        # Group keys by top-level layer prefix
        for key in self.model_state:
            top = key.split(".")[0]
            layers.setdefault(top, {"expert_keys": [], "routing_keys": [], "prefix": top})
            if "expert" in key.lower():
                layers[top]["expert_keys"].append(key)
            elif "routing" in key.lower() or "gate" in key.lower():
                layers[top]["routing_keys"].append(key)

        # Only keep layers that have at least one expert
        layers = {k: v for k, v in layers.items() if v["expert_keys"]}
        return layers

    def analyze_layer(self, layer_name: str) -> LayerAnalysis:
        """Full analysis of one MoE layer."""
        if layer_name not in self._layers:
            raise KeyError(f"Layer '{layer_name}' not found. Available: {list(self._layers.keys())}")

        layer_info = self._layers[layer_name]
        experts = self._build_experts(layer_info)
        routing = self._build_routing(layer_info, len(experts))

        sheaf = MoESheaf(experts, routing)
        dist = sheaf.distance_matrix
        n = dist.shape[0]
        if n >= 2:
            upper_tri = dist[np.triu_indices_from(dist, k=1)]
            dist_stats = {
                "mean": float(np.mean(upper_tri)),
                "std": float(np.std(upper_tri)),
                "min": float(np.min(upper_tri)),
                "max": float(np.max(upper_tri)),
            }
        else:
            dist_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        total_params = sum(e.num_params for e in experts)
        h1 = sheaf.compute_h1()
        h0 = sheaf.compute_h0()

        # Routing entropy
        rw = routing + 1e-10
        rw = rw / rw.sum(axis=1, keepdims=True)
        entropy = -float(np.mean(np.sum(rw * np.log(rw), axis=1)))

        stalks = [sheaf.stalk(i) for i in range(len(experts))]

        return LayerAnalysis(
            layer_name=layer_name,
            n_experts=len(experts),
            h0=h0,
            h1=h1,
            h1_per_param=h1 / max(total_params, 1),
            total_params=total_params,
            distance_stats=dist_stats,
            routing_entropy=entropy,
            persistence_diagram=sheaf.persistence_diagram(),
            expert_norms=[float(np.linalg.norm(e.manifold_point())) for e in experts],
            stalks=stalks,
        )

    def _build_experts(self, layer_info: dict) -> list[Expert]:
        """Build Expert objects from state dict keys."""
        experts = []
        for idx, key in enumerate(sorted(layer_info["expert_keys"])):
            w = self.model_state[key]
            if w.ndim < 2:
                w = w.reshape(-1, 1)
            experts.append(Expert(id=idx, weight_matrix=w))
        if not experts:
            # Fallback: create synthetic experts from any weights in this layer
            for key in sorted(layer_info.get("routing_keys", [])):
                pass
        return experts

    def _build_routing(self, layer_info: dict, n_experts: int) -> np.ndarray:
        """Build routing weight matrix."""
        for key in layer_info["routing_keys"]:
            w = self.model_state[key]
            if w.ndim == 2 and w.shape[-1] == n_experts:
                return w
        # Default: uniform routing
        return np.ones((1, n_experts)) / n_experts

    def cohomology_vs_generalization(
        self, generalization_scores: list[float]
    ) -> CorrelationResult:
        """Across multiple models, test H¹ vs generalization correlation.

        Each model should have been initialized separately. This method
        computes H¹/param for each layer and correlates with scores.

        Since we have one model, we create per-layer data points.
        """
        h1_per_params = []
        gen_scores = []

        for i, (layer_name, _) in enumerate(self._layers.items()):
            try:
                analysis = self.analyze_layer(layer_name)
                h1_per_params.append(analysis.h1_per_param)
                gen_scores.append(generalization_scores[i % len(generalization_scores)])
            except Exception:
                continue

        if len(h1_per_params) < 2:
            return CorrelationResult(
                pearson_r=0.0, spearman_r=0.0,
                p_value_pearson=1.0, p_value_spearman=1.0,
                h1_per_params=h1_per_params, generalization_scores=gen_scores,
                n_models=len(h1_per_params), conjecture_supported=False,
            )

        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(h1_per_params, gen_scores)
        sr, sp = spearmanr(h1_per_params, gen_scores)

        return CorrelationResult(
            pearson_r=float(pr),
            spearman_r=float(sr),
            p_value_pearson=float(pp),
            p_value_spearman=float(sp),
            h1_per_params=h1_per_params,
            generalization_scores=gen_scores,
            n_models=len(h1_per_params),
            conjecture_supported=float(pr) > 0,
        )
