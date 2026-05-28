"""moe-sheaf: Sheaf cohomology of MoE routing."""

from moe_sheaf.expert import Expert
from moe_sheaf.routing import MoESheaf
from moe_sheaf.cohomology import compute_h0, compute_h1, persistence_diagram
from moe_sheaf.analysis import MoEAnalysis, LayerAnalysis, CorrelationResult
from moe_sheaf.conjecture import evaluate_conjecture, evaluate_conjecture_across_models, ConjectureResult, evaluate_conjecture_across_models as test_conjecture_across_models

__all__ = [
    "Expert",
    "MoESheaf",
    "compute_h0",
    "compute_h1",
    "persistence_diagram",
    "MoEAnalysis",
    "LayerAnalysis",
    "CorrelationResult",
    "evaluate_conjecture",
    "evaluate_conjecture_across_models",
    "ConjectureResult",
]
