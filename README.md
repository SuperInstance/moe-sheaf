# moe-sheaf

**Sheaf cohomology of Mixture-of-Experts routing.**

This library implements and tests DeepSeek's novel conjecture: *the sheaf cohomology H¹ of an MoE routing layer encodes the model's generalization capacity per activated parameter.*

## The Conjecture

> Models whose MoE routing sheaf has higher first cohomology H¹ per activated parameter generalize better on unseen data.

H¹ measures "obstructions" — inconsistencies in how the routing sheaf glues together across the expert manifold. Higher H¹ means the experts cover more diverse, non-trivially overlapping regions of function space, which (conjecturally) yields better generalization.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from moe_sheaf import Expert, MoESheaf, test_conjecture

# Create synthetic experts
experts = [Expert(id=i, weight_matrix=np.random.randn(64, 32)) for i in range(8)]
routing = np.random.randn(100, 8).softmax(axis=1)  # 100 tokens, 8 experts

sheaf = MoESheaf(experts, routing)
print(f"H⁰ = {sheaf.compute_h0()}")   # connected components
print(f"H¹ = {sheaf.compute_h1():.4f}")  # obstruction

# Test the conjecture
result = test_conjecture(experts, routing, generalization_score=0.85)
print(f"Conjecture supported: {result.correlation_sign == 'positive'}")
```

## Architecture

- **`expert.py`** — Expert as a point on a weight manifold
- **`routing.py`** — MoE routing weights interpreted as a sheaf on the expert manifold
- **`cohomology.py`** — Computation of H⁰, H¹, and persistence diagrams via Vietoris-Rips filtration
- **`analysis.py`** — Full analysis pipeline for real model state dicts
- **`conjecture.py`** — Statistical test of DeepSeek's generalization-conjecture

## License

MIT
