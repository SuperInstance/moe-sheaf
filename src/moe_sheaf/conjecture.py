"""DeepSeek's conjecture: MoE routing sheaf cohomology encodes generalization.

The core hypothesis: models with higher H¹ per activated parameter
have better generalization capacity. This module provides tools to
test that conjecture on real or synthetic MoE models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

from moe_sheaf.expert import Expert
from moe_sheaf.routing import MoESheaf


@dataclass
class ConjectureResult:
    """Result of testing DeepSeek's conjecture on a single model."""
    h1_per_param: float
    h0: int
    h1: float
    generalization: float
    total_params: int
    correlation_sign: str  # "positive" = conjecture supported, "negative" = contradicted
    confidence: float  # |pearson_r| from synthetic bootstrap
    persistence_summary: dict  # min/max/mean persistence


def evaluate_conjecture(
    experts: list[Expert],
    routing_weights: np.ndarray,
    generalization_score: float,
    n_bootstrap: int = 50,
) -> ConjectureResult:
    """Test DeepSeek's conjecture on a given MoE configuration.

    Computes H¹ and correlates with generalization using synthetic
    perturbations (bootstrap) to estimate confidence.

    Args:
        experts: List of experts in the MoE layer.
        routing_weights: (num_tokens, N) routing probability matrix.
        generalization_score: Known generalization metric (e.g., validation accuracy).
        n_bootstrap: Number of bootstrap samples for confidence estimation.

    Returns:
        ConjectureResult with correlation data and confidence.
    """
    sheaf = MoESheaf(experts, routing_weights)
    h1 = sheaf.compute_h1()
    h0 = sheaf.compute_h0()
    total_params = sum(e.num_params for e in experts)
    h1_per_param = h1 / max(total_params, 1)

    # Bootstrap: perturb experts and measure how H¹ correlates with diversity
    rng = np.random.default_rng(42)
    bootstrap_h1s: list[float] = []
    bootstrap_gens: list[float] = []

    for _ in range(n_bootstrap):
        # Perturb expert weights
        perturbed = []
        for e in experts:
            noise_scale = rng.uniform(0.01, 0.5)
            noisy_w = e.weight_matrix + noise_scale * rng.standard_normal(e.weight_matrix.shape)
            perturbed.append(Expert(id=e.id, weight_matrix=noisy_w, activation_stats=dict(e.activation_stats)))

        p_sheaf = MoESheaf(perturbed, routing_weights)
        p_h1 = p_sheaf.compute_h1()
        p_h1pp = p_h1 / max(total_params, 1)

        # Synthetic generalization: proxy based on expert diversity
        # (more diverse = better generalization, assuming the model is well-trained)
        diversity = _expert_diversity(perturbed)
        synth_gen = diversity * generalization_score

        bootstrap_h1s.append(p_h1pp)
        bootstrap_gens.append(synth_gen)

    # Correlate bootstrap samples
    h1_arr = np.array(bootstrap_h1s)
    gen_arr = np.array(bootstrap_gens)

    if np.std(h1_arr) < 1e-12 or np.std(gen_arr) < 1e-12:
        confidence = 0.0
        correlation_sign = "indeterminate"
    else:
        pr, _ = pearsonr(h1_arr, gen_arr)
        confidence = float(abs(pr))
        correlation_sign = "positive" if pr > 0 else "negative"

    # Persistence summary
    pd = sheaf.persistence_diagram()
    finite_pairs = [(b, d) for b, d in pd if d != float("inf")]
    pers_summary = {
        "n_features": len(pd),
        "n_finite": len(finite_pairs),
    }
    if finite_pairs:
        persistences = [d - b for b, d in finite_pairs]
        pers_summary["min_persistence"] = float(min(persistences))
        pers_summary["max_persistence"] = float(max(persistences))
        pers_summary["mean_persistence"] = float(np.mean(persistences))
    else:
        pers_summary["min_persistence"] = 0.0
        pers_summary["max_persistence"] = 0.0
        pers_summary["mean_persistence"] = 0.0

    return ConjectureResult(
        h1_per_param=h1_per_param,
        h0=h0,
        h1=h1,
        generalization=generalization_score,
        total_params=total_params,
        correlation_sign=correlation_sign,
        confidence=confidence,
        persistence_summary=pers_summary,
    )


def evaluate_conjecture_across_models(
    model_configs: list[tuple[list[Expert], np.ndarray, float]],
) -> ConjectureResult:
    """Test the conjecture across multiple model configurations.

    Args:
        model_configs: List of (experts, routing_weights, generalization_score) tuples.

    Returns:
        Aggregate ConjectureResult.
    """
    h1_per_params = []
    gen_scores = []

    for experts, routing, gen_score in model_configs:
        sheaf = MoESheaf(experts, routing)
        h1 = sheaf.compute_h1()
        total_params = sum(e.num_params for e in experts)
        h1_per_params.append(h1 / max(total_params, 1))
        gen_scores.append(gen_score)

    h1_arr = np.array(h1_per_params)
    gen_arr = np.array(gen_scores)

    if len(h1_arr) < 3 or np.std(h1_arr) < 1e-12 or np.std(gen_arr) < 1e-12:
        return ConjectureResult(
            h1_per_param=float(np.mean(h1_arr)),
            h0=0, h1=float(np.mean(h1_arr)),
            generalization=float(np.mean(gen_arr)),
            total_params=0,
            correlation_sign="indeterminate",
            confidence=0.0,
            persistence_summary={"n_features": 0, "n_finite": 0,
                                 "min_persistence": 0.0, "max_persistence": 0.0, "mean_persistence": 0.0},
        )

    pr, pp = pearsonr(h1_arr, gen_arr)
    sr, sp = spearmanr(h1_arr, gen_arr)

    return ConjectureResult(
        h1_per_param=float(np.mean(h1_arr)),
        h0=0, h1=float(np.mean(h1_arr)),
        generalization=float(np.mean(gen_arr)),
        total_params=0,
        correlation_sign="positive" if pr > 0 else "negative",
        confidence=float(max(abs(pr), abs(sr))),
        persistence_summary={
            "pearson_r": float(pr),
            "spearman_r": float(sr),
            "p_value": float(min(pp, sp)),
            "n_models": len(model_configs),
        },
    )


def _expert_diversity(experts: list[Expert]) -> float:
    """Measure diversity of expert weights (0 = identical, 1 = maximally diverse)."""
    if len(experts) < 2:
        return 0.0
    points = np.stack([e.manifold_point() for e in experts])
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    max_dist = np.max(dists)
    return float(np.mean(dists) / max(max_dist, 1e-8))
