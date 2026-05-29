# moe-sheaf

**Sheaf cohomology of Mixture-of-Experts routing — test whether H¹ per parameter predicts generalization.**

Models each MoE layer as a cellular sheaf on the expert manifold. Experts are points in weight space; routing overlap defines restriction maps; Vietoris-Rips filtration on pairwise distances builds the topology. Computes persistent H⁰ (connected expert clusters) and H¹ (routing obstruction) to test DeepSeek's conjecture: higher H¹ per activated parameter → better generalization.

## What This Gives You

- **Expert manifold representation** — each expert as a point on its weight manifold with activation statistics
- **Sheaf construction** — stalks = expert weights, restriction maps = routing overlap
- **Persistent cohomology** — H⁰ and H¹ via Vietoris-Rips filtration
- **Conjecture testing** — correlates H¹/param with generalization using bootstrap confidence
- **Full analysis pipeline** — feed a model state dict, get layer-by-layer cohomology report
- **Correlation analysis** — Pearson and Spearman r across multiple models

## Quick Start

```python
from moe_sheaf import Expert, MoESheaf, compute_h0, compute_h1, evaluate_conjecture

# Define experts
experts = [Expert.random(id=i, input_dim=256, output_dim=64, seed=i) for i in range(8)]

# Routing weights: (num_tokens, n_experts)
routing = softmax(logits, axis=1)

# Build sheaf and compute cohomology
sheaf = MoESheaf(experts, routing)
h0 = compute_h0(sheaf.distance_matrix(), epsilon=5.0)
h1 = compute_h1(sheaf.distance_matrix(), sheaf.routing_overlap(), epsilon=5.0)

# Test DeepSeek conjecture
result = evaluate_conjecture(experts, routing, generalization_score=0.87)
print(f"H¹/param: {result.h1_per_param:.4f}, supported: {result.correlation_sign}")
```

## Installation

```bash
pip install -e .
```

Requires: `numpy>=1.24`, `scipy>=1.10`

## Testing

```bash
pytest tests/
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[persistent-sheaf](https://github.com/SuperInstance/persistent-sheaf)** — Rust persistent sheaf cohomology library
- **moe-sheaf** — Sheaf cohomology applied to MoE routing (this repo)

## License

MIT
