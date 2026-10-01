import pandas as pd

def final_comparison():
    data = {
        "Variant": [
            "Zero-Shot Baseline (Current)",
            "Few-Shot Fine-Tuned (Task 1)",
            "Reduced-Feature Zero-Shot (Task 2)",
            "Simple Model Zero-Shot (Task 3)",
            "Recalibrated Normalization (Task 4)"
        ],
        "Recall": [0.0000, 0.0000, 0.0015, 0.0185, 0.2694],
        "Precision": [0.0000, 0.0000, 0.0143, 1.0000, 0.2823],
        "F1": [0.0000, 0.0000, 0.0028, 0.0363, 0.2757],
        "FPR": [0.0154, 0.0097, 0.1062, 0.0000, 0.7037],
        "MCC": [-0.0880, -0.0704, -0.2317, 0.0965, -0.4344]
    }
    
    df = pd.DataFrame(data)
    print("\n--- Final Cross-Dataset Generalization Comparison ---")
    print(df.to_string(index=False))

if __name__ == '__main__':
    final_comparison()
