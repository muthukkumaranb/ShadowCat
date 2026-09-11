"""
Smoke tests for the ablation pipeline and LOEO harness using synthetic data.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from ml2.models.fusion import FusedModel, TemporalOnlyBaseline
from ml2.training.metrics import next_state_error, rollout_error, compute_all_rollout_errors
from ml2.data.dataset import SnapshotDataset, build_paired_sequences
from ml2.experiments.loeo import (
    LOEOHarness,
    Episode,
    create_episodes_from_graphs,
    default_metrics_fn,
)


def _make_synthetic_graphs(n=30, num_nodes=5, num_features=11):
    """Creates n synthetic graphs."""
    graphs = []
    for t in range(n):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        graphs.append(Data(x=x, edge_index=edge_index, timestamp=t))
    return graphs


class TestMetrics:
    def test_next_state_error_returns_float(self):
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        pairs = build_paired_sequences(ds)
        nse = next_state_error(model, pairs, z_dim=64, device="cpu")
        assert isinstance(nse, float)
        assert not np.isnan(nse)

    def test_next_state_error_empty_pairs(self):
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        nse = next_state_error(model, [], z_dim=64, device="cpu")
        assert nse == 0.0

    def test_rollout_error_returns_dict(self):
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        graphs = _make_synthetic_graphs(10)
        result = rollout_error(model, graphs, K=3, z_dim=64, device="cpu")
        assert "per_step_mse" in result
        assert "cumulative_mse" in result
        assert "mean_mse" in result
        assert len(result["per_step_mse"]) == 3

    def test_rollout_error_insufficient_data(self):
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        graphs = _make_synthetic_graphs(2)
        result = rollout_error(model, graphs, K=5, z_dim=64, device="cpu")
        assert np.isnan(result["mean_mse"])

    def test_compute_all_rollout_errors(self):
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        graphs = _make_synthetic_graphs(15)
        results = compute_all_rollout_errors(model, graphs, K_values=[1, 2, 3], z_dim=64)
        assert set(results.keys()) == {1, 2, 3}
        for K, r in results.items():
            assert "mean_mse" in r

    def test_temporal_only_metrics(self):
        """Temporal-only baseline should also produce valid metrics."""
        model = TemporalOnlyBaseline(z_dim=64, output_dim=11)
        graphs = _make_synthetic_graphs(10)
        ds = SnapshotDataset(graphs)
        pairs = build_paired_sequences(ds)
        nse = next_state_error(model, pairs, z_dim=64, device="cpu")
        assert isinstance(nse, float)


class TestLOEOHarness:
    def test_episode_creation(self):
        graphs = _make_synthetic_graphs(30)
        episodes = create_episodes_from_graphs(
            graphs, episode_boundaries=[0, 10, 20], episode_ids=["ep0", "ep1", "ep2"]
        )
        assert len(episodes) == 3
        assert episodes[0].episode_id == "ep0"
        assert len(episodes[0]) == 10
        assert len(episodes[2]) == 10

    def test_episode_creation_no_boundaries(self):
        graphs = _make_synthetic_graphs(20)
        episodes = create_episodes_from_graphs(graphs)
        assert len(episodes) == 1
        assert len(episodes[0]) == 20

    def test_harness_register_model(self):
        harness = LOEOHarness()
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        harness.register_model(model, "test_fused")
        assert "test_fused" in harness._models

    def test_harness_requires_minimum_episodes(self):
        harness = LOEOHarness()
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        harness.register_model(model, "test")

        episodes = [Episode(episode_id="ep0", data=_make_synthetic_graphs(5))]
        with pytest.raises(ValueError, match="≥2 episodes"):
            harness.run(episodes, lambda m, d, dev: {"mse": 0.0})

    def test_harness_run_and_summary(self):
        harness = LOEOHarness()
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        harness.register_model(model, "test_model")

        graphs = _make_synthetic_graphs(30)
        episodes = create_episodes_from_graphs(
            graphs, episode_boundaries=[0, 10, 20], episode_ids=["ep0", "ep1", "ep2"]
        )

        def simple_metrics(model, test_data, device):
            return {"dummy_mse": np.random.rand()}

        results = harness.run(episodes, simple_metrics, verbose=False)
        assert "test_model" in results
        assert len(results["test_model"]) == 3  # 3 folds

        summary = harness.summary()
        assert len(summary) == 4  # 3 folds + 1 aggregate
        assert "dummy_mse" in summary.columns

    def test_harness_reset(self):
        harness = LOEOHarness()
        model = FusedModel(node_feature_dim=11, graph_embedding_dim=64, z_dim=64, output_dim=11)
        harness.register_model(model, "test")
        harness._results["test"] = ["fake_result"]
        harness.reset()
        assert harness._results["test"] == []

    def test_harness_no_models_error(self):
        harness = LOEOHarness()
        episodes = [
            Episode(episode_id="ep0", data=[]),
            Episode(episode_id="ep1", data=[]),
        ]
        with pytest.raises(ValueError, match="No models"):
            harness.run(episodes, lambda m, d, dev: {})
