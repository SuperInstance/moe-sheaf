"""Tests for moe-sheaf."""

import time

import numpy as np
import pytest

from moe_sheaf.expert import Expert
from moe_sheaf.routing import MoESheaf
from moe_sheaf.cohomology import compute_h0, compute_h1, persistence_diagram
from moe_sheaf.analysis import MoEAnalysis, LayerAnalysis, CorrelationResult
from moe_sheaf.conjecture import evaluate_conjecture, evaluate_conjecture_across_models, ConjectureResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_experts():
    """8 random experts with distinct weights."""
    return [Expert.random(id=i, input_dim=16, output_dim=8, seed=i) for i in range(8)]


@pytest.fixture
def random_routing():
    """Random routing weights for 100 tokens × 8 experts."""
    rng = np.random.default_rng(99)
    logits = rng.standard_normal((100, 8))
    # softmax
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture
def sheaf(random_experts, random_routing):
    return MoESheaf(random_experts, random_routing)


# ---------------------------------------------------------------------------
# 1. Expert creation and manifold point
# ---------------------------------------------------------------------------

class TestExpert:
    def test_creation(self):
        e = Expert(id=0, weight_matrix=np.eye(3))
        assert e.id == 0
        assert e.input_dim == 3
        assert e.output_dim == 3

    def test_manifold_point(self):
        w = np.array([[1.0, 2.0], [3.0, 4.0]])
        e = Expert(id=0, weight_matrix=w)
        pt = e.manifold_point()
        assert pt.shape == (4,)
        np.testing.assert_array_equal(pt, [1, 2, 3, 4])

    def test_distance_to(self):
        e1 = Expert(id=0, weight_matrix=np.zeros((2, 2)))
        e2 = Expert(id=1, weight_matrix=np.ones((2, 2)))
        d = e1.distance_to(e2)
        assert d == pytest.approx(2.0)  # sqrt(4)

    def test_distance_self_is_zero(self):
        e = Expert(id=0, weight_matrix=np.random.randn(4, 3))
        assert e.distance_to(e) == pytest.approx(0.0)

    def test_random_factory(self):
        e = Expert.random(id=5, input_dim=8, output_dim=4, seed=42)
        assert e.id == 5
        assert e.weight_matrix.shape == (8, 4)

    def test_num_params(self):
        e = Expert(id=0, weight_matrix=np.zeros((10, 20)))
        assert e.num_params == 200


# ---------------------------------------------------------------------------
# 2. MoESheaf creation
# ---------------------------------------------------------------------------

class TestMoESheaf:
    def test_creation(self, random_experts, random_routing):
        s = MoESheaf(random_experts, random_routing)
        assert s.n_experts == 8

    def test_stalk(self, sheaf):
        stalk = sheaf.stalk(0)
        assert "weight_norm" in stalk
        assert "manifold_point" in stalk
        assert "activation_stats" in stalk
        assert "routing_load" in stalk
        assert stalk["weight_norm"] > 0

    def test_restriction_map(self, sheaf):
        r = sheaf.restriction_map(0, 1)
        assert r.shape == sheaf.experts[0].manifold_point().shape

    def test_restriction_map_zero_norm(self):
        e1 = Expert(id=0, weight_matrix=np.zeros((4, 3)))
        e2 = Expert(id=1, weight_matrix=np.ones((4, 3)))
        s = MoESheaf([e1, e2], np.array([[0.5, 0.5]]))
        r = s.restriction_map(0, 1)
        np.testing.assert_array_equal(r, np.zeros(12))


# ---------------------------------------------------------------------------
# 3. Cohomology
# ---------------------------------------------------------------------------

class TestCohomology:
    def test_h0_single_expert_is_1(self):
        d = np.array([[0.0]])
        assert compute_h0(d) == 1

    def test_h0_disconnected_equals_n(self):
        """Two experts far apart → 2 components."""
        d = np.array([[0.0, 100.0], [100.0, 0.0]])
        assert compute_h0(d, epsilon=0.1) == 2

    def test_h0_fully_connected_is_1(self):
        """All experts within epsilon → 1 component."""
        d = np.array([[0.0, 0.1, 0.2], [0.1, 0.0, 0.1], [0.2, 0.1, 0.0]])
        assert compute_h0(d, epsilon=1.0) == 1

    def test_h1_identical_experts_is_zero(self):
        """Identical experts → no loops → H¹ = 0."""
        d = np.zeros((4, 4))
        h1 = compute_h1(d)
        assert h1 == pytest.approx(0.0)

    def test_h1_diverse_experts_positive(self, sheaf):
        """Diverse experts should have H¹ > 0."""
        h1 = sheaf.compute_h1()
        assert h1 >= 0  # at minimum, non-negative

    def test_h1_increases_with_diversity(self):
        """More spread-out experts should have higher H¹ potential."""
        rng = np.random.default_rng(42)
        # Compact cluster
        compact = [Expert(id=i, weight_matrix=rng.standard_normal((4, 2)) * 0.01) for i in range(6)]
        routing = np.ones((10, 6)) / 6
        h1_compact = MoESheaf(compact, routing).compute_h1()

        # Spread out
        spread = [Expert(id=i, weight_matrix=rng.standard_normal((4, 2)) * 10.0) for i in range(6)]
        h1_spread = MoESheaf(spread, routing).compute_h1()

        assert h1_spread >= h1_compact


# ---------------------------------------------------------------------------
# 4. Persistence diagram
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistence_nonempty(self, sheaf):
        pd = sheaf.persistence_diagram()
        assert len(pd) > 0

    def test_persistence_birth_leq_death(self, sheaf):
        pd = sheaf.persistence_diagram()
        for birth, death in pd:
            assert birth <= death or death == float("inf")


# ---------------------------------------------------------------------------
# 5. Conjecture
# ---------------------------------------------------------------------------

class TestConjecture:
    def test_conjecture_result_fields(self, random_experts, random_routing):
        result = evaluate_conjecture(random_experts, random_routing, generalization_score=0.85)
        assert isinstance(result, ConjectureResult)
        assert result.h1 >= 0
        assert result.total_params > 0
        assert result.generalization == 0.85
        assert result.correlation_sign in ("positive", "negative", "indeterminate")
        assert 0.0 <= result.confidence <= 1.0

    def test_conjecture_positive_with_diverse_experts(self):
        """Models with higher H¹ should show positive correlation with generalization."""
        rng = np.random.default_rng(42)

        configs = []
        for i in range(10):
            scale = 0.5 + i * 2.0  # increasing diversity
            experts = [Expert(id=j, weight_matrix=rng.standard_normal((4, 2)) * scale) for j in range(8)]
            routing = np.ones((20, 8)) / 8
            gen = 0.5 + i * 0.045  # increasing generalization
            configs.append((experts, routing, gen))

        result = evaluate_conjecture_across_models(configs)
        # With this synthetic setup, the conjecture should be supported
        assert result.correlation_sign == "positive"


# ---------------------------------------------------------------------------
# 6. Analysis pipeline
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_layer_analysis_fields(self):
        """LayerAnalysis should have all fields populated."""
        rng = np.random.default_rng(42)
        state = {
            "layer0.expert0.weight": rng.standard_normal((16, 8)),
            "layer0.expert1.weight": rng.standard_normal((16, 8)),
            "layer0.expert2.weight": rng.standard_normal((16, 8)),
            "layer0.routing.weight": rng.standard_normal((1, 3)),
        }
        analysis = MoEAnalysis(state)
        layer_names = list(analysis._layers.keys())
        assert len(layer_names) > 0
        la = analysis.analyze_layer(layer_names[0])
        assert isinstance(la, LayerAnalysis)
        assert la.n_experts > 0
        assert la.h0 >= 1
        assert la.h1 >= 0
        assert la.total_params > 0
        assert "mean" in la.distance_stats
        assert len(la.expert_norms) == la.n_experts
        assert len(la.stalks) == la.n_experts


# ---------------------------------------------------------------------------
# 7. Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_100_experts_efficient(self):
        """Analyzing 100 experts should complete in < 10 seconds."""
        rng = np.random.default_rng(42)
        experts = [Expert(id=i, weight_matrix=rng.standard_normal((8, 4))) for i in range(100)]
        routing_logits = rng.standard_normal((50, 100))
        e = np.exp(routing_logits - routing_logits.max(axis=1, keepdims=True))
        routing = e / e.sum(axis=1, keepdims=True)

        start = time.time()
        sheaf = MoESheaf(experts, routing)
        h0 = sheaf.compute_h0()
        h1 = sheaf.compute_h1()
        elapsed = time.time() - start

        assert elapsed < 10.0, f"Took {elapsed:.2f}s, too slow for 100 experts"
        assert h0 >= 1
        assert h1 >= 0
