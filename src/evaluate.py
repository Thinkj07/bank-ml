import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ROOT, load_config


def evaluate(model, X, y):
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
    }


def log_result(name, metrics):
    path = ROOT / load_config()["paths"]["results"]
    row = pd.DataFrame([{"model": name, **metrics}])
    if path.exists() and path.stat().st_size > 0:
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)
