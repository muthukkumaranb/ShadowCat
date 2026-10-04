"""Proper scoring rules and calibration diagnostics for Part P (P-t3 / P-t4).

All functions return PER-WINDOW values (averaged over dimensions where relevant) so that
episode-level bootstrap confidence intervals can be computed afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

Z50 = norm.ppf(0.75)
Z90 = norm.ppf(0.95)
SQRT_PI = np.sqrt(np.pi)


# ----------------------------------------------------------------------------- Gaussian forecasts
def gaussian_scores(mean: np.ndarray, sd: np.ndarray, y: np.ndarray) -> dict:
    """Closed-form CRPS, NLL, interval coverage, 90% width and PIT for diagonal Gaussians. Shapes (n, d)."""
    z = (y - mean) / sd
    crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / SQRT_PI)
    nll = 0.5 * np.log(2 * np.pi) + np.log(sd) + 0.5 * z ** 2
    return {
        "crps": crps.mean(1),
        "nll": nll.mean(1),
        "cov50": (np.abs(z) <= Z50).mean(1),
        "cov90": (np.abs(z) <= Z90).mean(1),
        "width90": (2 * Z90 * sd).mean(1),
        "pit": norm.cdf(z),  # (n, d)
    }


def gaussian_samples(mean: np.ndarray, sd: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw m samples per window from a diagonal Gaussian: (n, m, d). Used only for the energy score."""
    eps = rng.standard_normal((mean.shape[0], m, mean.shape[1]))
    return mean[:, None, :] + sd[:, None, :] * eps


# ----------------------------------------------------------------------------- Monte Carlo forecasts
def sample_crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS from an ensemble, E|X-y| - 0.5 E|X-X'| (V-statistic). samples (n, m, d), y (n, d) -> (n,)."""
    n, m, d = samples.shape
    term1 = np.abs(samples - y[:, None, :]).mean(1)
    xs = np.sort(samples, axis=1)
    i = np.arange(1, m + 1)[None, :, None]
    term2 = (2 * (2 * i - m - 1) * xs).sum(1) / (m * m)  # = E|X-X'|
    return (term1 - 0.5 * term2).mean(1)


def sample_scores(samples: np.ndarray, y: np.ndarray) -> dict:
    mu = samples.mean(1)
    sd = samples.std(1) + 1e-8
    z = (y - mu) / sd
    q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95], axis=1)
    return {
        "crps": sample_crps(samples, y),
        "nll": (0.5 * np.log(2 * np.pi) + np.log(sd) + 0.5 * z ** 2).mean(1),  # Gaussian approximation
        "cov50": ((y >= q25) & (y <= q75)).mean(1),
        "cov90": ((y >= q05) & (y <= q95)).mean(1),
        "width90": (q95 - q05).mean(1),
        "pit": (samples < y[:, None, :]).mean(1),
    }


def energy_score(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multivariate energy score E||X-y|| - 0.5 E||X-X'||. samples (n, m, d) -> (n,)."""
    t1 = np.linalg.norm(samples - y[:, None, :], axis=2).mean(1)
    diff = samples[:, :, None, :] - samples[:, None, :, :]
    t2 = np.linalg.norm(diff, axis=3).mean((1, 2))
    return t1 - 0.5 * t2


def pit_histogram(pit: np.ndarray, bins: int = 10) -> list:
    return np.histogram(pit.ravel(), bins=bins, range=(0, 1))[0].astype(int).tolist()


# ----------------------------------------------------------------------------- binary
def brier(p, y):
    return (p - y) ** 2


def logloss(p, y, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        out.append({"bin": [round(edges[b], 2), round(edges[b + 1], 2)], "n": int(mask.sum()),
                    "mean_pred": float(p[mask].mean()) if mask.any() else None,
                    "frac_pos": float(y[mask].mean()) if mask.any() else None})
    return out


# ----------------------------------------------------------------------------- bootstrap
def episode_bootstrap_diff(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Paired bootstrap over episodes (groups) of mean(a) - mean(b). Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([idx_by_g[g] for g in pick])
        diffs.append(a[ix].mean() - b[ix].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)
