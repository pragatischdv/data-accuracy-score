import numpy as np
from sklearn.metrics import r2_score

def das_classification(pred_original: np.ndarray, pred_synthetic: np.ndarray) -> float:
    """Percentage agreement between two prediction arrays."""
    if len(pred_original) != len(pred_synthetic):
        raise ValueError("Predictions must have the same length.")
    matches = np.sum(pred_original == pred_synthetic)
    return (matches / len(pred_original)) * 100

def das_regression(y_true: np.ndarray, y_pred_original: np.ndarray, y_pred_synthetic: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    DAS for regression:
    ((R2_original - R2_synthetic) / R2_original) * 100
    Returns: (das_value, r2_original, r2_synthetic)
    """
    r2_orig = r2_score(y_true, y_pred_original)
    r2_synth = r2_score(y_true, y_pred_synthetic)
    denom = r2_orig if abs(r2_orig) > eps else eps
    das_value = ((r2_orig - r2_synth) / denom) * 100
    return das_value, r2_orig, r2_synth
