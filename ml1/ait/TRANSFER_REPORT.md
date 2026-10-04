# AIT-LDS Zero-Shot Transfer & Finetuning (G-t4)

## 1. Zero-Shot Performance (Pre-Trained on CIC-IDS2018)
The pretrained `LSTMGaussianWorldModel` was evaluated directly on `russellmitchell` without any training updates on AIT data.
- **Zero-Shot MSE**: 0.1175
- **Zero-Shot MAE**: 0.2187

## 2. Fine-Tuning (1 Epoch)
The model was fine-tuned for 1 epoch with a learning rate of `1e-4`.
- **Fine-Tuned MSE**: 0.0825
- **Fine-Tuned MAE**: 0.1506

## 3. Analysis
Due to the vast structural differences between CIC-IDS2018 and AIT-LDS v2.0, Zero-Shot transfer is expected to yield higher error rates, primarily because AIT-LDS data largely contains zero-imputed values (masks) resulting from the lack of whole-network telemetry.
Fine-tuning for even just 1 epoch drastically minimizes this error as the network adjusts to the missing domains (e.g. Traffic Volume & Flow Timing groups being consistently `0.0`).
