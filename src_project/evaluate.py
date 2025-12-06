import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def eval_classifier(y_true, y_pred):
    """Returns simple classification metrics."""

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Fallback if ROC AUC cannot be computed
        roc_auc = 0.0

    return {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall Score": recall,
        "ROC AUC Score": roc_auc
    }


def eval_regressor(y_true, y_pred):
    """Return simple regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }