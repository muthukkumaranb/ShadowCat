import os
import sys
import numpy as np
from pathlib import Path
import json

workspace_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from backend.conformal import SplitConformalPredictor
from backend.predict import ShadowcatPipeline

def main():
    print("Evaluating Conformal Coverage (Per-step vs Joint)")
    pipeline = ShadowcatPipeline()
    # It auto-loads calibration residuals inside _get_conformal_calibration_data
    # We will use those residuals as our "held out" validation set for measurement
    # Actually, conformal calibration uses the val set to calibrate. To measure coverage properly,
    # we should use a test set. But we only have pooled val residuals available easily.
    # We can measure empirical coverage on the same set (should be exact due to conformal guarantees)
    # or split the residuals into cal and test.
    
    preds, targets, source = pipeline._get_conformal_calibration_data()
    n = len(preds)
    # Split into 50% cal, 50% test for empirical measurement
    np.random.seed(42)
    indices = np.random.permutation(n)
    cal_idx = indices[:n//2]
    test_idx = indices[n//2:]
    
    cal_preds = preds[cal_idx]
    cal_targets = targets[cal_idx]
    test_preds = preds[test_idx]
    test_targets = targets[test_idx]
    
    predictor = SplitConformalPredictor(coverage=0.90, aci_mode=False)
    predictor.calibrate(cal_preds, cal_targets)
    
    # Measure marginal coverage
    covered_marginal = 0
    for p, y in zip(test_preds, test_targets):
        lb, ub = predictor.predict_interval(p)
        if lb <= y <= ub:
            covered_marginal += 1
            
    # Measure joint coverage
    covered_joint = 0
    # For joint coverage, a trajectory of K=5 steps is covered if ALL 5 steps fall inside.
    # We will group test samples into chunks of 5 to simulate trajectories.
    num_trajectories = len(test_preds) // 5
    for i in range(num_trajectories):
        traj_preds = test_preds[i*5 : (i+1)*5]
        traj_targets = test_targets[i*5 : (i+1)*5]
        
        # Marginal joint coverage (using normal intervals for all 5)
        all_covered_marginal = True
        for p, y in zip(traj_preds, traj_targets):
            lb, ub = predictor.predict_interval(p)
            if not (lb <= y <= ub):
                all_covered_marginal = False
                break
                
        # Bonferroni joint coverage
        all_covered_bonf = True
        for p, y in zip(traj_preds, traj_targets):
            lb, ub = predictor.predict_interval_joint(p)
            if not (lb <= y <= ub):
                all_covered_bonf = False
                break
                
        if all_covered_bonf:
            covered_joint += 1
            
    # Wait, the prompt says "measure both per-step and joint coverage on held-out data".
    # I should compute the empirical coverage percentage.
    
    marginal_cov_pct = covered_marginal / len(test_preds)
    joint_cov_marginal_pct = 0 # Using marginal intervals for joint trajectory (will be lower than 90%)
    
    # Calculate real joint coverage for trajectories using marginal intervals
    traj_covered_marginal = 0
    traj_covered_bonf = 0
    for i in range(num_trajectories):
        traj_preds = test_preds[i*5 : (i+1)*5]
        traj_targets = test_targets[i*5 : (i+1)*5]
        
        ok_marg = True
        ok_bonf = True
        for p, y in zip(traj_preds, traj_targets):
            lb, ub = predictor.predict_interval(p)
            lb_b, ub_b = predictor.predict_interval_joint(p)
            if not (lb <= y <= ub): ok_marg = False
            if not (lb_b <= y <= ub_b): ok_bonf = False
            
        if ok_marg: traj_covered_marginal += 1
        if ok_bonf: traj_covered_bonf += 1
        
    print(f"Target Per-Step Coverage: 90.0%")
    print(f"Empirical Per-Step Coverage: {marginal_cov_pct*100:.2f}%")
    print(f"Trajectory Coverage using Marginal Intervals (Expected ~59%): {traj_covered_marginal/num_trajectories*100:.2f}%")
    print(f"Trajectory Coverage using Joint Intervals (Bonferroni): {traj_covered_bonf/num_trajectories*100:.2f}%")

if __name__ == '__main__':
    main()
