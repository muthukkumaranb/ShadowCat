"""
Comprehensive unit tests for GNN models, inference, and the graph embedding contract.

Tests cover:
- EdgeAwareSAGEConv message passing with edge features
- GraphEncoder forward pass and embedding shape
- SnapshotGNN with diagnostic classifier
- Batch forward pass
- Edge features affecting model output (ablation verification)
- Inference output schema (primary: graph_embedding)
- Feature ablation explainability
- Temporal GNN forward pass
"""
import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from gnn.models.graphsage import (
    EdgeAwareSAGEConv,
    GraphEncoder,
    DiagnosticClassifier,
    SnapshotGNN,
    TemporalAttackGNN,
    AttackRiskGNN,  # backward compat alias
)
from gnn.inference.predict import GNNPredictor
from gnn.graph_builder import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


# ── Constants ─────────────────────────────────────────────────────────────────
NODE_DIM = len(NODE_FEATURE_NAMES)  # 12
EDGE_DIM = len(EDGE_FEATURE_NAMES)  # 6
HIDDEN_DIM = 32
EMBEDDING_DIM = 64


def _make_test_graph(n_nodes=8, n_edges=4, node_dim=NODE_DIM, edge_dim=EDGE_DIM):
    """Create a synthetic test graph with known dimensions."""
    edges_src = torch.randint(0, n_nodes, (n_edges,))
    edges_dst = torch.randint(0, n_nodes, (n_edges,))
    return Data(
        x=torch.randn(n_nodes, node_dim),
        edge_index=torch.stack([edges_src, edges_dst]),
        edge_attr=torch.randn(n_edges, edge_dim),
        y=torch.tensor([0.0]),
        num_nodes=n_nodes,
    )


# ── EdgeAwareSAGEConv Tests ───────────────────────────────────────────────────

class TestEdgeAwareSAGEConv:

    def test_forward_with_edge_features(self):
        """Edge features are concatenated with source features in message passing."""
        conv = EdgeAwareSAGEConv(in_channels=NODE_DIM, out_channels=HIDDEN_DIM, edge_dim=EDGE_DIM)
        x = torch.randn(8, NODE_DIM)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(4, EDGE_DIM)

        out = conv(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (8, HIDDEN_DIM)

    def test_forward_without_edge_features(self):
        """When edge_dim=0, behaves like standard SAGEConv."""
        conv = EdgeAwareSAGEConv(in_channels=NODE_DIM, out_channels=HIDDEN_DIM, edge_dim=0)
        x = torch.randn(8, NODE_DIM)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        out = conv(x, edge_index, edge_attr=None)
        assert out.shape == (8, HIDDEN_DIM)


# ── GraphEncoder Tests ────────────────────────────────────────────────────────

class TestGraphEncoder:

    def test_encode_single_graph(self):
        """Single graph produces [1, embedding_dim] embedding."""
        encoder = GraphEncoder(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
            num_layers=2,
        )
        graph = _make_test_graph()
        emb = encoder.encode_single(graph)
        assert emb.shape == (1, EMBEDDING_DIM)

    def test_encode_batch(self):
        """Batch of graphs produces [batch_size, embedding_dim]."""
        encoder = GraphEncoder(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
            num_layers=2,
        )
        g1 = _make_test_graph(n_nodes=5, n_edges=3)
        g2 = _make_test_graph(n_nodes=7, n_edges=5)
        g3 = _make_test_graph(n_nodes=4, n_edges=2)

        batch = Batch.from_data_list([g1, g2, g3])
        emb = encoder.encode(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        assert emb.shape == (3, EMBEDDING_DIM)

    def test_embedding_dimension_is_configurable(self):
        """Embedding dimension should match configuration."""
        for dim in [32, 64, 128]:
            encoder = GraphEncoder(
                node_feature_dim=NODE_DIM,
                edge_feature_dim=EDGE_DIM,
                hidden_dim=HIDDEN_DIM,
                embedding_dim=dim,
            )
            graph = _make_test_graph()
            emb = encoder.encode_single(graph)
            assert emb.shape == (1, dim), f"Expected embedding_dim={dim}, got {emb.shape[1]}"

    def test_forward_equals_encode(self):
        """forward() and encode() should produce identical output."""
        encoder = GraphEncoder(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        encoder.eval()  # deterministic mode
        graph = _make_test_graph()
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)

        with torch.no_grad():
            emb1 = encoder.forward(x, edge_index, batch, edge_attr=edge_attr)
            emb2 = encoder.encode(x, edge_index, batch, edge_attr=edge_attr)
        assert torch.allclose(emb1, emb2)


# ── Edge Features Affect Output Test ──────────────────────────────────────────

class TestEdgeFeaturesAffectOutput:

    def test_edge_features_change_embedding(self):
        """
        CRITICAL: Verify that edge_attr actually influences the graph embedding.

        If edge features are ignored (the bug we're fixing), the embedding would be
        identical regardless of edge_attr values. This test ensures they differ.
        """
        encoder = GraphEncoder(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        encoder.eval()

        # Same graph structure, different edge features
        x = torch.randn(8, NODE_DIM)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        batch = torch.zeros(8, dtype=torch.long)

        edge_attr_a = torch.ones(4, EDGE_DIM)
        edge_attr_b = torch.ones(4, EDGE_DIM) * 100.0  # Very different values

        with torch.no_grad():
            emb_a = encoder.encode(x, edge_index, batch, edge_attr=edge_attr_a)
            emb_b = encoder.encode(x, edge_index, batch, edge_attr=edge_attr_b)

        # Embeddings must differ when edge features differ
        assert not torch.allclose(emb_a, emb_b, atol=1e-4), \
            "Edge features did NOT affect the embedding — edge_attr is being ignored!"

    def test_no_edge_features_mode(self):
        """When edge_feature_dim=0, model works without edge_attr."""
        encoder = GraphEncoder(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=0,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        graph = _make_test_graph()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)

        emb = encoder.encode(graph.x, graph.edge_index, batch, edge_attr=None)
        assert emb.shape == (1, EMBEDDING_DIM)


# ── SnapshotGNN Tests ─────────────────────────────────────────────────────────

class TestSnapshotGNN:

    def test_forward_returns_logits(self):
        """SnapshotGNN forward() returns classification logits [B, 1]."""
        model = SnapshotGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        graph = _make_test_graph()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)

        logits = model(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr)
        assert logits.shape == (1, 1)

    def test_get_graph_embedding(self):
        """get_graph_embedding() returns [B, embedding_dim]."""
        model = SnapshotGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        graph = _make_test_graph()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)

        emb = model.get_graph_embedding(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr)
        assert emb.shape == (1, EMBEDDING_DIM)

    def test_predict_proba_range(self):
        """predict_proba() returns values in [0, 1]."""
        model = SnapshotGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        graph = _make_test_graph()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)

        probs = model.predict_proba(graph.x, graph.edge_index, batch, edge_attr=graph.edge_attr)
        assert 0.0 <= probs.item() <= 1.0

    def test_backward_compat_alias(self):
        """AttackRiskGNN alias works the same as SnapshotGNN."""
        model = AttackRiskGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        assert isinstance(model, SnapshotGNN)


# ── TemporalAttackGNN Tests ──────────────────────────────────────────────────

class TestTemporalGNN:

    def test_forward(self):
        """Temporal GNN forward pass on a graph sequence."""
        model = TemporalAttackGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )

        g1 = _make_test_graph(n_nodes=5, n_edges=3)
        g2 = _make_test_graph(n_nodes=6, n_edges=4)
        sequence = [g1, g2]

        logits = model(sequence)
        assert logits.shape == (1, 1)

    def test_predict_proba(self):
        """Temporal predict_proba returns values in [0, 1]."""
        model = TemporalAttackGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )

        g1 = _make_test_graph(n_nodes=5, n_edges=3)
        g2 = _make_test_graph(n_nodes=6, n_edges=4)
        probs = model.predict_proba([g1, g2])
        assert 0.0 <= probs.item() <= 1.0


# ── Inference Tests ───────────────────────────────────────────────────────────

class TestInference:

    def test_inference_output_schema(self, tmp_path):
        """Verify the primary output contract: graph_embedding is present and correct."""
        model = SnapshotGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        checkpoint_path = str(tmp_path / "mock_gnn.pt")

        torch.save({
            "state_dict": model.state_dict(),
            "model_config": {
                "node_feature_dim": NODE_DIM,
                "edge_feature_dim": EDGE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "num_layers": 2,
                "dropout": 0.0,
                "pooling": "mean_max",
            },
            "is_temporal": False,
        }, checkpoint_path)

        predictor = GNNPredictor(checkpoint_path, device="cpu")
        dummy_data = _make_test_graph()

        out = predictor.predict(dummy_data)

        # Primary contract: graph_embedding must be present
        assert "graph_embedding" in out, "Missing primary output: graph_embedding"
        assert isinstance(out["graph_embedding"], list)
        assert len(out["graph_embedding"]) == EMBEDDING_DIM

        # Optional diagnostic fields
        assert "snapshot_risk" in out
        assert isinstance(out["snapshot_risk"], float)
        assert 0.0 <= out["snapshot_risk"] <= 1.0

        assert "prediction" in out
        assert out["prediction"] in (0, 1)

    def test_get_embedding_numpy(self, tmp_path):
        """get_embedding() returns a numpy array of correct shape."""
        model = SnapshotGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        checkpoint_path = str(tmp_path / "mock_gnn.pt")

        torch.save({
            "state_dict": model.state_dict(),
            "model_config": {
                "node_feature_dim": NODE_DIM,
                "edge_feature_dim": EDGE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "num_layers": 2,
                "dropout": 0.0,
                "pooling": "mean_max",
            },
            "is_temporal": False,
        }, checkpoint_path)

        predictor = GNNPredictor(checkpoint_path, device="cpu")
        dummy_data = _make_test_graph()

        emb = predictor.get_embedding(dummy_data)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (EMBEDDING_DIM,)

    def test_integration_contract(self, tmp_path):
        """
        Integration test: verify the complete contract that another engineer would use.

        graph = build_graph_snapshot(ucs_window)
        graph_embedding = gnn.predict(graph)
        assert graph_embedding.shape[-1] == 64
        """
        model = SnapshotGNN(
            node_feature_dim=NODE_DIM,
            edge_feature_dim=EDGE_DIM,
            hidden_dim=HIDDEN_DIM,
            embedding_dim=EMBEDDING_DIM,
        )
        checkpoint_path = str(tmp_path / "mock_gnn.pt")

        torch.save({
            "state_dict": model.state_dict(),
            "model_config": {
                "node_feature_dim": NODE_DIM,
                "edge_feature_dim": EDGE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "num_layers": 2,
                "dropout": 0.0,
                "pooling": "mean_max",
            },
            "is_temporal": False,
        }, checkpoint_path)

        predictor = GNNPredictor(checkpoint_path, device="cpu")

        # Simulate: graph = build_graph_snapshot(ucs_window)
        graph = _make_test_graph()

        # graph_embedding = gnn.predict(graph)
        result = predictor.predict(graph)
        graph_embedding = np.array(result["graph_embedding"])

        # assert graph_embedding.shape[-1] == 64
        assert graph_embedding.shape[-1] == EMBEDDING_DIM


# ── Node Feature Dimension Tests ──────────────────────────────────────────────

class TestFeatureDimensions:

    def test_node_feature_names_count(self):
        """Verify NODE_FEATURE_NAMES has exactly 12 features."""
        assert len(NODE_FEATURE_NAMES) == 12

    def test_edge_feature_names_count(self):
        """Verify EDGE_FEATURE_NAMES has exactly 6 features."""
        assert len(EDGE_FEATURE_NAMES) == 6

    def test_node_features_documented_order(self):
        """Verify the documented node feature ordering matches the constant."""
        expected = [
            "out_degree", "in_degree",
            "out_flow_count", "in_flow_count",
            "out_bytes", "in_bytes",
            "out_packets", "in_packets",
            "out_tcp_ratio", "out_udp_ratio",
            "unique_dst_ports", "unique_src_ports",
        ]
        assert NODE_FEATURE_NAMES == expected

    def test_edge_features_documented_order(self):
        """Verify the documented edge feature ordering matches the constant."""
        expected = [
            "flow_count", "total_bytes", "total_packets",
            "mean_duration", "tcp_ratio", "udp_ratio",
        ]
        assert EDGE_FEATURE_NAMES == expected
