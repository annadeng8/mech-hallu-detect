import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def summarize_logs(logs):
    tiers = [x["tier"] for x in logs]
    lat = [x["latency_ms"] for x in logs]
    return {
        "tier1_frac": np.mean(np.array(tiers) == 1),
        "tier2_frac": np.mean(np.array(tiers) == 2),
        "tier3_frac": np.mean(np.array(tiers) == 3),
        "mean_latency_ms": np.mean(lat),
    }

