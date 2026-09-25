import pandas as pd

g_det = pd.read_csv('ml1/artifacts/lstm/lstm_graph_detection_loeo_folds.csv')
g_ons = pd.read_csv('ml1/artifacts/lstm/lstm_graph_onset_loeo_folds.csv')
p_det = pd.read_csv('ml1/artifacts/lstm/lstm_detection_loeo_folds.csv')
p_ons = pd.read_csv('ml1/artifacts/lstm/lstm_onset_loeo_folds.csv')

print("=== PLAIN STACKED LSTM (Current Production) ===")
print(f"Detection - F1: {p_det['f1'].mean():.4f}, Prec: {p_det['precision'].mean():.4f}, Rec: {p_det['recall'].mean():.4f}, FPR: {p_det['fpr'].mean():.4f}")
print(f"Onset     - F1: {p_ons['f1'].mean():.4f}, Prec: {p_ons['precision'].mean():.4f}, Rec: {p_ons['recall'].mean():.4f}, FPR: {p_ons['fpr'].mean():.4f}")

print("\n=== GRAPH-AUGMENTED STACKED LSTM ===")
print(f"Detection - F1: {g_det['f1'].mean():.4f}, Prec: {g_det['precision'].mean():.4f}, Rec: {g_det['recall'].mean():.4f}, FPR: {g_det['fpr'].mean():.4f}")
print(f"Onset     - F1: {g_ons['f1'].mean():.4f}, Prec: {g_ons['precision'].mean():.4f}, Rec: {g_ons['recall'].mean():.4f}, FPR: {g_ons['fpr'].mean():.4f}")
