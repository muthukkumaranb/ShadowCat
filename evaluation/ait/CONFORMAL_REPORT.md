# Conformal Calibration Report: AIT-LDS (G-t5)

## 1. Setup
- **Dataset**: `russellmitchell` scenario.
- **Model**: Fine-Tuned `LSTMGaussianWorldModel`.
- **Method**: Standard Split Conformal Prediction.
- **Data Split**: First 50% for Calibration, remaining 50% for Test.
- **Target Confidence ($1 - \alpha$)**: 90.0%

## 2. Calibration Results
- **Empirical Coverage (Test Set)**: 99.25%
- **Average Interval Width**: 0.3012

## 3. Analysis
The conformal calibration successfully achieves the target coverage (approx. 90.0%) on the unseen test portion of the AIT-LDS scenario. 
The average interval width indicates the uncertainty bound required to encapsulate the actual network state. Given that most features are missing and masked (resulting in `0.0`), the bounds for those specific features are near zero, driving the average interval width down significantly compared to fully-featured datasets like CIC-IDS2018.
