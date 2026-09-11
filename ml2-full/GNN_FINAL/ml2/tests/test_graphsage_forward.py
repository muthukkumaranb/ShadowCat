"""
Smoke tests for GraphSAGE forward pass and fusion models.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from ml2.models.graphsage import GraphSAGEEncoder, GraphBranch
from ml2.models.fusion import FusedModel, TemporalOnlyBaseline, GraphOnlyBaseline


def _make_random_graph(num_nodes=10, num_features=11, num_edges=20):
    """Creates a random graph for testing."""
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)


class TestGraphSAGEEncoder:
    def test_output_shape(self):
        encoder = GraphSAGEEncoder(in_channels=11, hidden_dim=64, embedding_dim=64, num_layers=2)
        g = _make_random_graph()
        h = encoder(g.x, g.edge_index)
        assert h.shape == (10, 64)

    def test_single_layer(self):
        encoder = GraphSAGEEncoder(in_channels=11, hidden_dim=64, embedding_dim=32, num_layers=1)
        g = _make_random_graph()
        h = encoder(g.x, g.edge_index)
        assert h.shape == (10, 32)

    def test_three_layers(self):
        encoder = GraphSAGEEncoder(in_channels=11, hidden_dim=64, embedding_dim=32, num_layers=3)
        g = _make_random_graph()
        h = encoder(g.x, g.edge_index)
        assert h.shape == (10, 32)

    def test_gradient_flow(self):
        """Verify gradients flow from output back to input."""
        encoder = GraphSAGEEncoder(in_channels=11, hidden_dim=64, embedding_dim=64, num_layers=2)
        g = _make_random_graph()
        g.x.requires_grad_(True)

        h = encoder(g.x, g.edge_index)
        loss = h.sum()
        loss.backward()

        assert g.x.grad is not None
        assert not torch.all(g.x.grad == 0)


class TestGraphBranch:
    def test_output_shapes(self):
        branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
        g = _make_random_graph()
        h_v, g_emb = branch(g.x, g.edge_index)
        assert h_v.shape == (10, 64)  # node embeddings
        assert g_emb.shape == (1, 64)  # pooled graph embedding

    def test_with_batch(self):
        branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
        # Two graphs batched together
        x = torch.randn(15, 11)  # 10 + 5 nodes
        edge_index = torch.randint(0, 15, (2, 30))
        batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(5, dtype=torch.long)])
        h_v, g_emb = branch(x, edge_index, batch)
        assert h_v.shape == (15, 64)
        assert g_emb.shape == (2, 64)  # 2 graphs


class TestFusedModel:
    def test_forward_shape(self):
        model = FusedModel(
            node_feature_dim=11, graph_hidden_dim=64,
            graph_embedding_dim=64, z_dim=64, output_dim=11,
        )
        g = _make_random_graph()
        z_t = torch.randn(1, 64)
        out = model(g.x, g.edge_index, z_t)
        assert out.shape == (1, 11)

    def test_gradient_flow(self):
        model = FusedModel(
            node_feature_dim=11, graph_hidden_dim=64,
            graph_embedding_dim=64, z_dim=64, output_dim=11,
        )
        g = _make_random_graph()
        z_t = torch.randn(1, 64, requires_grad=True)
        out = model(g.x, g.edge_index, z_t)
        loss = out.sum()
        loss.backward()
        assert z_t.grad is not None

    def test_model_type(self):
        model = FusedModel()
        assert model.model_type == "fused"


class TestTemporalOnlyBaseline:
    def test_forward_shape(self):
        model = TemporalOnlyBaseline(z_dim=64, output_dim=11)
        g = _make_random_graph()
        z_t = torch.randn(1, 64)
        out = model(g.x, g.edge_index, z_t)
        assert out.shape == (1, 11)

    def test_graph_invariance(self):
        """Output should be the same regardless of graph input."""
        model = TemporalOnlyBaseline(z_dim=64, output_dim=11)
        model.eval()
        z_t = torch.randn(1, 64)
        g1 = _make_random_graph(num_nodes=5)
        g2 = _make_random_graph(num_nodes=20)
        out1 = model(g1.x, g1.edge_index, z_t)
        out2 = model(g2.x, g2.edge_index, z_t)
        assert torch.allclose(out1, out2)

    def test_model_type(self):
        model = TemporalOnlyBaseline()
        assert model.model_type == "temporal_only"


class TestGraphOnlyBaseline:
    def test_forward_shape(self):
        model = GraphOnlyBaseline(
            node_feature_dim=11, graph_hidden_dim=64,
            graph_embedding_dim=64, output_dim=11,
        )
        g = _make_random_graph()
        z_t = torch.randn(1, 64)
        out = model(g.x, g.edge_index, z_t)
        assert out.shape == (1, 11)

    def test_z_invariance(self):
        """Output should be the same regardless of z_t input."""
        model = GraphOnlyBaseline(
            node_feature_dim=11, graph_hidden_dim=64,
            graph_embedding_dim=64, output_dim=11,
        )
        model.eval()
        g = _make_random_graph()
        z1 = torch.randn(1, 64)
        z2 = torch.randn(1, 64)
        out1 = model(g.x, g.edge_index, z1)
        out2 = model(g.x, g.edge_index, z2)
        assert torch.allclose(out1, out2)

    def test_model_type(self):
        model = GraphOnlyBaseline()
        assert model.model_type == "graph_only"
