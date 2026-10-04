"""Sanity tests for the Part P scoring rules (no data needed)."""
import numpy as np

from evaluation.prob_forecast import scoring as S
from ml1.prob_forecast.forecasters import DiagonalAR1, inflate


def test_sample_crps_matches_closed_form_gaussian():
    rng = np.random.default_rng(0)
    mu, sd = np.zeros((50, 3)), np.ones((50, 3)) * 2.0
    y = rng.normal(0, 2, size=(50, 3))
    closed = S.gaussian_scores(mu, sd, y)["crps"]
    samples = S.gaussian_samples(mu, sd, 4000, rng)
    mc = S.sample_crps(samples, y)
    assert np.allclose(closed, mc, atol=0.05)


def test_crps_prefers_the_true_distribution():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, size=(2000, 1))
    good = S.gaussian_scores(np.zeros_like(y), np.ones_like(y), y)["crps"].mean()
    too_wide = S.gaussian_scores(np.zeros_like(y), 3 * np.ones_like(y), y)["crps"].mean()
    too_narrow = S.gaussian_scores(np.zeros_like(y), 0.3 * np.ones_like(y), y)["crps"].mean()
    assert good < too_wide and good < too_narrow


def test_coverage_of_calibrated_gaussian():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, size=(20000, 1))
    sc = S.gaussian_scores(np.zeros_like(y), np.ones_like(y), y)
    assert abs(sc["cov90"].mean() - 0.90) < 0.01 and abs(sc["cov50"].mean() - 0.50) < 0.01


def test_inflate_keeps_mean_and_scales_spread():
    rng = np.random.default_rng(3)
    s = rng.normal(5, 1, size=(4, 64, 2))
    out = inflate(s, 2.0)
    assert np.allclose(out.mean(1), s.mean(1))
    assert np.allclose(out.std(1), 2 * s.std(1))


def test_episode_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 10)
    a = rng.normal(0, 1, 300)
    pt, lo, hi = S.episode_bootstrap_diff(a - 1.0, a, groups, n_boot=500)
    assert hi < 0 and abs(pt + 1.0) < 1e-9


def test_ar1_recovers_coefficient():
    rng = np.random.default_rng(5)
    x = np.zeros((3000, 1))
    for t in range(1, 3000):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    seg = np.zeros(3000, int)
    idx = np.arange(3000)
    m = DiagonalAR1().fit(x, idx, seg, set(idx.tolist()))
    assert abs(m.a[0] - 0.7) < 0.05
