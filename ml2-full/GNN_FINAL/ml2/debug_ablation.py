"""
ML2 Ablation Bug Investigation -- Post-Fix Validation Script (Windows-safe)
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from ml2.models.fusion import FusedModel
from ml2.models.graphsage import GraphBranch
from ml2.training.metrics import next_state_error


def make_random_graph(num_nodes=20, num_edges=40, feat_dim=11):
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)


def get_fixed_target(graph, device="cpu"):
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    return global_mean_pool(graph.x, batch)


def train_model_on_fixed_targets(model, pairs, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for gt, gtp1 in pairs:
            batch_t = torch.zeros(gt.x.size(0), dtype=torch.long)
            z_t = torch.zeros(1, 64)
            pred = model(gt.x, gt.edge_index, z_t, batch_t)
            
            # Use FIXED target space
            target = get_fixed_target(gtp1)
            
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_noise_substitution():
    print("=" * 70)
    print("POST-FIX VALIDATION: Noise Substitution Test (Multiple Seeds) + Rollout")
    print("=" * 70)

    from ml2.training.metrics import rollout_error
    
    seeds = [42, 123, 999]
    real_mses = []
    noise_mses = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n--- RUNNING SEED {seed} ---")
        
        graphs = [make_random_graph(num_nodes=np.random.randint(10, 50)) for _ in range(40)]
        pairs = [(graphs[i], graphs[i + 1]) for i in range(len(graphs) - 1)]

        # 1. Train real FusedModel
        real_model = FusedModel(node_feature_dim=11, z_dim=64, output_dim=11)
        train_model_on_fixed_targets(real_model, pairs)

        # 2. Train NoiseFusedModel
        class NoiseFusedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_branch = GraphBranch(in_channels=11, hidden_dim=64, embedding_dim=64)
                self.projection = torch.nn.Sequential(
                    torch.nn.Linear(128, 128),
                    torch.nn.LayerNorm(128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.LayerNorm(128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 11),
                )
                self.rollout_projection = torch.nn.Sequential(
                    torch.nn.Linear(11 + 64, 128),
                    torch.nn.LayerNorm(128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.LayerNorm(128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 11),
                )

            def forward(self, x, edge_index, z_t, batch=None):
                g_t = torch.randn(1, 64)  # RANDOM NOISE instead of graph embedding
                fused = torch.cat([g_t, z_t], dim=-1)
                return self.projection(fused)

        noise_model = NoiseFusedModel()
        train_model_on_fixed_targets(noise_model, pairs)

        # 3. Evaluate both models
        real_model.eval()
        noise_model.eval()
        
        real_mse = next_state_error(real_model, pairs, z_dim=64, device="cpu")
        noise_mse = next_state_error(noise_model, pairs, z_dim=64, device="cpu")
        
        real_mses.append(real_mse)
        noise_mses.append(noise_mse)
        
        print(f"  Real-graph MSE:       {real_mse:.8f}")
        print(f"  Noise-substituted MSE: {noise_mse:.8f}")

        # Check rollout curve for real model on this seed
        print("  Rollout curve (Real Model):")
        for k in [1, 2, 3, 5, 10]:
            res = rollout_error(real_model, graphs, K=k, z_dim=64, device="cpu")
            print(f"    K={k:2d}: {res['mean_mse']:.8f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS (Averaged across seeds)")
    print("=" * 70)
    avg_real = np.mean(real_mses)
    avg_noise = np.mean(noise_mses)
    print(f"  Avg Real-graph MSE:       {avg_real:.8f} ± {np.std(real_mses):.8f}")
    print(f"  Avg Noise-substituted MSE: {avg_noise:.8f} ± {np.std(noise_mses):.8f}")
    print(f"  Avg Difference:           {avg_noise - avg_real:.8f}")

    # 4. Sanity-check the fixed target space itself
    targets = []
    for _, gtp1 in pairs:
        targets.append(get_fixed_target(gtp1).numpy().flatten())
    targets = np.array(targets)
    target_var = np.var(targets, axis=0)
    print("\nTARGET SPACE VARIANCE (11-dim raw features):")
    for i, var in enumerate(target_var):
        print(f"  Feature {i:2d}: var={var:.6f}")
    print(f"  Mean variance: {target_var.mean():.6f}")

    # Interpretation
    print("\nVERDICT:")
    if noise_mse - real_mse > 0.05 * noise_mse:
        print("  >>> PASS: Real-graph MSE is clearly lower than noise MSE.")
        print("  >>> The graph branch's structural content is now contributing.")
    elif abs(noise_mse - real_mse) < 0.01 * noise_mse:
        print("  >>> FAIL: Real and noise MSE are still comparable.")
        print("  >>> A third bug is likely present.")
    else:
        print("  >>> AMBIGUOUS: Real-graph is noticeably but not dramatically better.")
        print("  >>> Need to investigate further.")


if __name__ == "__main__":
    check_noise_substitution()
