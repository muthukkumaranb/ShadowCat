"""
Inference module for GNN graph representation encoder.

Primary Output Contract:
{
    "graph_embedding": [...]    # 64-D graph representation vector
}

Optional diagnostic output (if classifier head exists):
{
    "graph_embedding": [...],
    "snapshot_risk": float      # diagnostic P(attack) — NOT the World Model's final output
}

The graph_embedding is the GNN component's key deliverable for downstream
Fusion with LSTM temporal embeddings in the World Model pipeline.
"""
import os
import sys
import json
import argparse
from typing import List, Dict, Union, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from gnn.models.graphsage import GraphEncoder, SnapshotGNN, TemporalAttackGNN
from gnn.graph_builder import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES


class GNNPredictor:
    """
    Production inference class for the GNN Graph Representation Encoder.

    Primary interface:
        result = predictor.predict(graph)
        graph_embedding = result["graph_embedding"]  # 64-D list

    The predictor loads a trained checkpoint and produces deterministic
    graph embeddings for downstream consumption by the Fusion/LSTM system.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto", threshold: float = 0.5):
        """
        Args:
            checkpoint_path: Path to .pt checkpoint file containing state_dict + metadata
            device: 'cuda', 'cpu', or 'auto'
            threshold: Decision threshold for optional diagnostic prediction (default: 0.5)
        """
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.metadata = self.checkpoint.get("metadata", {})
        self.model_config = self.checkpoint.get("model_config", {})
        self.is_temporal = self.checkpoint.get("is_temporal", False)

        # Extract dimensions from checkpoint metadata
        node_feature_dim = self.model_config.get("node_feature_dim",
                            self.model_config.get("in_channels", len(NODE_FEATURE_NAMES)))
        edge_feature_dim = self.model_config.get("edge_feature_dim", len(EDGE_FEATURE_NAMES))
        hidden_dim = self.model_config.get("hidden_dim",
                      self.model_config.get("hidden_channels", 64))
        embedding_dim = self.model_config.get("embedding_dim", 64)
        num_layers = self.model_config.get("num_layers", 2)
        dropout = self.model_config.get("dropout", 0.3)
        pooling = self.model_config.get("pooling", "mean_max")

        if self.is_temporal:
            temporal_method = self.model_config.get("temporal_method", "mean")
            self.model = TemporalAttackGNN(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                temporal_method=temporal_method,
            )
        else:
            self.model = SnapshotGNN(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
            )

        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = embedding_dim

    @torch.no_grad()
    def predict(self, graph_input: Union[Data, List[Data]]) -> Dict[str, Any]:
        """
        Run inference on a single graph snapshot or a sequence of graph snapshots.

        Args:
            graph_input: Single PyG Data object or List of PyG Data objects

        Returns:
            Dict with primary output:
            {
                "graph_embedding": List[float],       # embedding_dim-D representation
                "snapshot_risk": float (optional),     # diagnostic P(attack)
                "prediction": int (optional),          # diagnostic binary (0/1)
            }
        """
        if isinstance(graph_input, list):
            # Sequence of graphs
            if self.is_temporal:
                seq_device = [g.to(self.device) for g in graph_input]
                temporal_emb = self.model.get_sequence_embedding(seq_device)
                logits = self.model.temporal_classifier(temporal_emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = temporal_emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]
            else:
                # Use encoder on the latest graph in sequence
                latest_graph = graph_input[-1].to(self.device)
                emb = self.model.encoder.encode_single(latest_graph)
                logits = self.model.classifier(emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]
        else:
            # Single graph
            single_graph = graph_input.to(self.device)
            if self.is_temporal:
                seq_device = [single_graph]
                temporal_emb = self.model.get_sequence_embedding(seq_device)
                logits = self.model.temporal_classifier(temporal_emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = temporal_emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]
            else:
                emb = self.model.encoder.encode_single(single_graph)
                logits = self.model.classifier(emb)
                risk_score = float(torch.sigmoid(logits).item())
                emb_list = emb.cpu().squeeze().tolist()
                if not isinstance(emb_list, list):
                    emb_list = [emb_list]

        prediction = 1 if risk_score >= self.threshold else 0

        return {
            "graph_embedding": [float(np.round(v, 6)) for v in emb_list],
            "snapshot_risk": float(np.round(risk_score, 6)),
            "prediction": int(prediction),
        }

    @torch.no_grad()
    def get_embedding(self, graph_input: Union[Data, List[Data]]) -> np.ndarray:
        """
        Convenience method: get just the graph embedding as a numpy array.

        Args:
            graph_input: Single PyG Data object

        Returns:
            numpy array of shape [embedding_dim]
        """
        result = self.predict(graph_input)
        return np.array(result["graph_embedding"], dtype=np.float32)

    @torch.no_grad()
    def explain_features(self, graph_input: Data, top_k: int = 5) -> Dict[str, Any]:
        """
        Feature ablation explainability (MVP approach per spec §28).

        Computes the embedding change when each node/edge feature is zeroed out.
        Features that cause the largest embedding shift are most influential.

        Args:
            graph_input: Single PyG Data object
            top_k: Number of top influential features to return

        Returns:
            Dict with feature importance rankings
        """
        single_graph = graph_input.to(self.device)

        # Baseline embedding
        baseline_emb = self.model.encoder.encode_single(single_graph).cpu().squeeze()

        # Node feature ablation
        node_importances = {}
        for i, name in enumerate(NODE_FEATURE_NAMES):
            ablated = single_graph.clone()
            ablated.x = ablated.x.clone()
            ablated.x[:, i] = 0.0
            ablated_emb = self.model.encoder.encode_single(ablated).cpu().squeeze()
            shift = float(torch.norm(baseline_emb - ablated_emb).item())
            node_importances[name] = shift

        # Edge feature ablation
        edge_importances = {}
        if single_graph.edge_attr is not None and single_graph.edge_attr.shape[1] > 0:
            for i, name in enumerate(EDGE_FEATURE_NAMES):
                if i < single_graph.edge_attr.shape[1]:
                    ablated = single_graph.clone()
                    ablated.edge_attr = ablated.edge_attr.clone()
                    ablated.edge_attr[:, i] = 0.0
                    ablated_emb = self.model.encoder.encode_single(ablated).cpu().squeeze()
                    shift = float(torch.norm(baseline_emb - ablated_emb).item())
                    edge_importances[name] = shift

        # Rank and select top-k
        all_importances = {f"node:{k}": v for k, v in node_importances.items()}
        all_importances.update({f"edge:{k}": v for k, v in edge_importances.items()})

        sorted_features = sorted(all_importances.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_k]

        return {
            "top_influential_features": [{"feature": f, "importance": round(v, 6)} for f, v in top_features],
            "node_feature_importances": {k: round(v, 6) for k, v in node_importances.items()},
            "edge_feature_importances": {k: round(v, 6) for k, v in edge_importances.items()},
        }


def run_cli():
    """Command-line inference entry point."""
    parser = argparse.ArgumentParser(description="GNN Graph Representation Encoder — Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_gnn.pt checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda, cpu, auto)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Diagnostic decision threshold")
    parser.add_argument("--explain", action="store_true", help="Run feature ablation explainability")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = GNNPredictor(args.checkpoint, device=args.device, threshold=args.threshold)

    # Generate synthetic dummy graph for verification
    dummy_data = Data(
        x=torch.randn(10, len(NODE_FEATURE_NAMES), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
        edge_attr=torch.randn(5, len(EDGE_FEATURE_NAMES), dtype=torch.float32),
        num_nodes=10,
    )

    result = predictor.predict(dummy_data)
    print("\n--- Inference Output ---")
    print(json.dumps(result, indent=2))

    assert len(result["graph_embedding"]) == predictor.embedding_dim, \
        f"Embedding dim mismatch: {len(result['graph_embedding'])} != {predictor.embedding_dim}"
    print(f"\n[OK] Graph embedding dimension verified: {len(result['graph_embedding'])}")

    if args.explain:
        print("\n--- Feature Ablation Explainability ---")
        explanation = predictor.explain_features(dummy_data, top_k=5)
        print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    run_cli()
