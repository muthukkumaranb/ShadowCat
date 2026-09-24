"""
Temporal Attribution Engine (TimeSHAP / Temporal Integrated Gradients)
Decomposes recurrent neural network predictions across both time steps and features.
Runs 100% offline with zero network dependencies.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class TemporalAttributor:
    """
    Computes cell-level, timestep-level, and feature-level attributions
    for sequential models (e.g. LSTM) using Temporal Integrated Gradients / Path Attribution.
    """

    def __init__(self, steps: int = 20):
        self.steps = steps

    def attribute(
        self,
        model: nn.Module,
        x_sequence: np.ndarray,
        feature_names: List[str],
        baseline: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Decomposes prediction of x_sequence into:
        - timestep_attributions: contribution of each lookback step t in [-(L-1)..0]
        - feature_attributions: total contribution of each feature across time
        - cell_matrix: (timesteps, features) attribution grid
        - top_temporal_events: ranked list of specific (timestep, feature) driver events
        """
        if device is None:
            device = torch.device("cpu")

        model = model.to(device)
        model.eval()

        seq = np.asarray(x_sequence, dtype=np.float32)
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)  # (1, T, D)
        elif seq.ndim != 3 or seq.shape[0] != 1:
            raise ValueError(f"Expected shape (1, T, D) or (T, D), got {seq.shape}")

        T = seq.shape[1]
        D = seq.shape[2]

        if baseline is None:
            baseline = np.zeros_like(seq)
        else:
            baseline = np.asarray(baseline, dtype=np.float32)
            if baseline.ndim == 2:
                baseline = np.expand_dims(baseline, axis=0)

        # Generate interpolated path: shape (steps, T, D)
        alphas = np.linspace(0.0, 1.0, self.steps, endpoint=True)
        interpolated = np.zeros((self.steps, T, D), dtype=np.float32)
        for i, a in enumerate(alphas):
            interpolated[i] = baseline[0] + a * (seq[0] - baseline[0])

        interp_tensor = torch.tensor(interpolated, dtype=torch.float32, device=device, requires_grad=True)

        # Forward pass through model
        if hasattr(model, "forward_logits"):
            outputs = model.forward_logits(interp_tensor)
        elif hasattr(model, "predict_next_state"):
            pred_state = model.predict_next_state(interp_tensor)
            # Use mean of predicted state vector as attribution target
            outputs = pred_state.mean(dim=-1)
        else:
            out = model(interp_tensor)
            if isinstance(out, tuple):
                outputs = out[0].mean(dim=-1)
            else:
                outputs = out.mean(dim=-1) if out.ndim > 1 else out

        # Backward pass to get path gradients
        loss = outputs.sum()
        loss.backward()

        grads = interp_tensor.grad.detach().cpu().numpy()  # (steps, T, D)
        avg_grads = np.mean(grads, axis=0)  # (T, D)
        diff = seq[0] - baseline[0]  # (T, D)

        # Attributions matrix (T, D)
        cell_attributions = diff * avg_grads

        # Positive / absolute impact
        abs_cells = np.abs(cell_attributions)
        total_mass = float(np.sum(abs_cells)) + 1e-8

        # 1. Timestep attributions (vector of length T)
        timestep_mass = np.sum(abs_cells, axis=1)  # (T,)
        timestep_relative = timestep_mass / total_mass

        timestep_records = []
        for t_idx in range(T):
            offset = t_idx - (T - 1)
            label = "Current Window (t)" if offset == 0 else f"Window t{offset}"
            timestep_records.append({
                "step_index": t_idx,
                "step_offset": int(offset),
                "step_label": label,
                "importance": round(float(timestep_relative[t_idx]), 4),
            })

        # 2. Feature attributions across time
        feature_mass = np.sum(abs_cells, axis=0)  # (D,)
        feature_relative = feature_mass / total_mass
        sorted_feat_idx = np.argsort(feature_relative)[::-1]

        feature_records = []
        for f_idx in sorted_feat_idx:
            fname = feature_names[f_idx] if f_idx < len(feature_names) else f"Feature_{f_idx}"
            feature_records.append({
                "feature": fname,
                "importance": round(float(feature_relative[f_idx]), 4),
            })

        # 3. Top discrete temporal driver events
        flat_indices = np.argsort(abs_cells.ravel())[::-1][:8]
        top_events = []
        for flat_idx in flat_indices:
            t_idx, f_idx = np.unravel_index(flat_idx, abs_cells.shape)
            offset = t_idx - (T - 1)
            t_label = "Current Window (t)" if offset == 0 else f"Window t{offset}"
            fname = feature_names[f_idx] if f_idx < len(feature_names) else f"Feature_{f_idx}"
            raw_val = float(seq[0, t_idx, f_idx])
            cell_contrib = float(cell_attributions[t_idx, f_idx])
            top_events.append({
                "time_label": t_label,
                "feature": fname,
                "feature_value": round(raw_val, 4),
                "contribution": round(cell_contrib, 4),
                "direction": "Elevates Risk" if cell_contrib > 0 else "Attenuates Risk",
            })

        return {
            "timesteps": timestep_records,
            "features": feature_records[:10],
            "top_temporal_events": top_events,
            "temporal_dimension": T,
            "feature_dimension": D,
            "attribution_method": "Temporal Integrated Gradients (TimeSHAP Equivalent)",
            "is_offline": True,
        }
