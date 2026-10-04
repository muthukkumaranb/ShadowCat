import pytest
import numpy as np
from backend.predict import ShadowcatPipeline

def test_current_stage_missing():
    backend = ShadowcatPipeline()
    
    # Mock detection threshold to guarantee it passes the threshold
    backend.detection_threshold_5pct_fpr = 0.5
    
    # Force family classifier to be empty
    backend.family_classifier_models = []
    backend.family_loeo_macro_f1 = 0.99
    
    # Dummy window of length 406
    dummy_window = np.zeros(406).tolist()
    
    # Make prediction where detection probability is above threshold
    res = backend._predict_current_stage(dummy_window, detection_prob=0.9)
    
    assert res["is_attack_detected"] is True
    assert res["current_stage"] == "not available"
    assert res["current_stage_confidence"] is None
    assert res["current_stage_family"] is None
    assert res["current_stage_tactic_id"] is None
    assert res["current_stage_url"] is None
