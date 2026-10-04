import numpy as np
import torch
import pytest
from backend.conformal import SplitConformalPredictor
from backend.models import MIMOStageClassificationHead

def test_conformal_joint_intervals_width():
    """
    Instantiates SplitConformalPredictor with aci_mode=False and asserts that 
    joint (Bonferroni) intervals are always >= as wide as the corresponding marginal intervals.
    """
    predictor = SplitConformalPredictor(coverage=0.90, aci_mode=False)
    
    # Mock calibration
    from numpy.random import uniform, randint
    val_preds = uniform(0, 1, 100)
    val_targets = randint(0, 2, 100)
    predictor.calibrate(val_preds, val_targets)
    
    # Assert marginal vs joint width
    point_prob = 0.5
    lb_m, ub_m = predictor.predict_interval(point_prob)
    lb_j, ub_j = predictor.predict_interval_joint(point_prob)
    
    width_m = ub_m - lb_m
    width_j = ub_j - lb_j
    
    assert width_j >= width_m, "Joint intervals (Bonferroni) should be at least as wide as marginal intervals."
    assert predictor.alpha_joint < predictor.alpha, "Bonferroni alpha should be smaller."

def test_conformal_aci_gamma_scaling():
    """
    Instantiates it with aci_mode=True and asserts that the per-horizon gamma values used 
    are indeed larger for horizon indices corresponding to K=4/K=5 than K=1-3.
    """
    predictor = SplitConformalPredictor(coverage=0.90, aci_mode=True)
    
    gammas = predictor.aci_gammas
    assert len(gammas) == 5, "Expected 5 gammas for K=1..5"
    
    # K=1..3 (indices 0..2), K=4..5 (indices 3..4)
    for idx_early in [0, 1, 2]:
        for idx_late in [3, 4]:
            assert gammas[idx_late] > gammas[idx_early], (
                f"Gamma for late horizon {idx_late} ({gammas[idx_late]}) "
                f"must be strictly greater than gamma for early horizon {idx_early} ({gammas[idx_early]})."
            )

def test_mimo_head_output_shape():
    """
    A smoke test that MIMOStageClassificationHead produces output of the expected shape 
    (5 horizons x num_classes) from a dummy 64-D latent input.
    """
    batch_size = 2
    latent_dim = 64
    num_classes = 6
    horizons = 5
    
    mimo_head = MIMOStageClassificationHead(
        latent_dim=latent_dim,
        num_classes=num_classes,
        horizons=horizons
    )
    
    dummy_input = torch.randn(batch_size, latent_dim)
    logits = mimo_head(dummy_input)
    
    expected_shape = (batch_size, horizons, num_classes)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
