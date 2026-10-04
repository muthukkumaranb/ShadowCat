"""Probabilistic K-step forecasters for Part P (P-t2).

Every forecaster returns, for a set of origin windows t and a horizon K, a predictive
distribution over the PCA-32 state S(t+K):
  - Gaussian forecasters return (mean, std), each of shape (n, d)
  - Monte Carlo forecasters return samples of shape (n, m, d)

All parameters are estimated on TRAINING windows of the current LOEO fold only.
"""
from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------------- helpers
def valid_pairs(indices: np.ndarray, segment: np.ndarray, k: int, allowed: set) -> np.ndarray:
    """Return origins t (from `indices`) such that t and t+k are both in `allowed` and in the same segment."""
    n = len(segment)
    out = [t for t in indices if t + k < n and segment[t] == segment[t + k] and (t + k) in allowed and t in allowed]
    return np.asarray(out, dtype=int)


# ----------------------------------------------------------------------------- F0
class Climatology:
    """F0: N(training mean, training per-dimension variance)."""

    name = "F0_climatology"

    def fit(self, P_train: np.ndarray):
        self.mu = P_train.mean(axis=0)
        self.sd = P_train.std(axis=0) + 1e-8
        return self

    def predict(self, P_origin: np.ndarray, k: int):
        n = len(P_origin)
        return np.tile(self.mu, (n, 1)), np.tile(self.sd, (n, 1))


# ----------------------------------------------------------------------------- F1 / F2
class NoisedPersistence:
    """F1: N(S(t), Sigma_K), Sigma_K = diagonal empirical variance of S(t+K) - S(t) on training pairs."""

    name = "F1_noised_persistence"

    def fit(self, P: np.ndarray, train_idx: np.ndarray, segment: np.ndarray, train_set: set, ks=range(1, 6)):
        self.sd = {}
        for k in ks:
            t = valid_pairs(train_idx, segment, k, train_set)
            d = P[t + k] - P[t]
            self.sd[k] = d.std(axis=0) + 1e-8
        return self

    def predict(self, P_origin: np.ndarray, k: int):
        return P_origin.copy(), np.tile(self.sd[k], (len(P_origin), 1))


class HeteroscedasticPersistence:
    """F2: like F1, but Sigma_K is estimated separately for training windows with label_binary(t) = 0 / 1.

    At test time the variance is chosen by the CURRENT window's stacked-detection decision
    (probability >= 5%-FPR threshold), never by the true label.
    """

    name = "F2_noised_persistence_hetero"

    def fit(self, P, train_idx, segment, train_set, labels, ks=range(1, 6)):
        self.sd = {}
        for k in ks:
            t = valid_pairs(train_idx, segment, k, train_set)
            for state in (0, 1):
                tt = t[labels[t] == state]
                d = P[tt + k] - P[tt]
                self.sd[(k, state)] = d.std(axis=0) + 1e-8
        return self

    def predict(self, P_origin, k, detection_decision):
        sd = np.stack([self.sd[(k, int(s))] for s in detection_decision])
        return P_origin.copy(), sd


# ----------------------------------------------------------------------------- F3
class DiagonalAR1:
    """F3: per-dimension AR(1) S_j(t+1) = a_j S_j(t) + b_j + e_j, with closed-form K-step mean and variance."""

    name = "F3_ar1"

    def fit(self, P, train_idx, segment, train_set):
        t = valid_pairs(train_idx, segment, 1, train_set)
        x, y = P[t], P[t + 1]
        xm, ym = x.mean(0), y.mean(0)
        var_x = ((x - xm) ** 2).mean(0) + 1e-12
        self.a = np.clip(((x - xm) * (y - ym)).mean(0) / var_x, -0.999, 0.999)
        self.b = ym - self.a * xm
        resid = y - (self.a * x + self.b)
        self.s2 = resid.var(0) + 1e-12
        return self

    def predict(self, P_origin, k):
        a, b, s2 = self.a, self.b, self.s2
        ak = a ** k
        mean = ak * P_origin + b * (1 - ak) / (1 - a)
        var = s2 * (1 - a ** (2 * k)) / (1 - a ** 2)
        return mean, np.tile(np.sqrt(var), (len(P_origin), 1))


# ----------------------------------------------------------------------------- F4 / F5
def build_sequences(P: np.ndarray, origins: np.ndarray, lookback: int = 30) -> np.ndarray:
    return np.stack([P[t - lookback + 1 : t + 1] for t in origins]).astype(np.float32)


@torch.no_grad()
def mc_rollout(model, P: np.ndarray, origins: np.ndarray, k_max: int = 5, n_samples: int = 64, seed: int = 42,
               lookback: int = 30, batch: int = 64) -> np.ndarray:
    """F4: autoregressive Monte Carlo rollout. Returns samples of shape (n, n_samples, k_max, d).

    At each step the next state is SAMPLED from the predicted diagonal Gaussian and fed back.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    seqs = build_sequences(P, origins, lookback)
    n, _, d = seqs.shape
    out = np.empty((n, n_samples, k_max, d), dtype=np.float32)
    for s in range(0, n, batch):
        x = torch.as_tensor(seqs[s : s + batch])
        b = x.shape[0]
        x = x.repeat_interleave(n_samples, dim=0)  # (b*m, L, d)
        for k in range(k_max):
            mean, std = model(x)
            nxt = mean + std * torch.randn(mean.shape, generator=gen)
            out[s : s + b, :, k, :] = nxt.view(b, n_samples, d).numpy()
            x = torch.cat([x[:, 1:, :], nxt.unsqueeze(1)], dim=1)
    return out


def inflate(samples: np.ndarray, c: float) -> np.ndarray:
    """F5 recalibration: scale the spread of the samples around their sample mean by c."""
    m = samples.mean(axis=1, keepdims=True)
    return m + c * (samples - m)
