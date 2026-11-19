import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import ROOT
from src.data import load_train_test
from src.models import gb, rf
from src.tuning import SEED

RESULT_DIR = ROOT / "reports" / "results"
N_ROUNDS = 1000


def read_json(filename):
    with open(RESULT_DIR / filename, encoding="utf-8") as file:
        return json.load(file)


def summarize(metric, name, values):
    values = np.array(values)
    return {
        "metric": metric,
        "name": name,
        "mean": values.mean(),
        "low_2.5": np.percentile(values, 2.5),
        "high_97.5": np.percentile(values, 97.5),
    }


def main():
    X_train, X_test, y_train, y_test = load_train_test()
    rf_info = read_json("rf_best_params.json")
    gb_info = read_json("gb_best_params.json")

    rf_model = rf.build(**rf_info["best_params"]).fit(X_train, y_train)
    gb_model = gb.build(**gb_info["best_params"]).fit(X_train, y_train)

    rf_score = rf_model.predict_proba(X_test)[:, 1]
    gb_score = gb_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_score >= rf_info["threshold"]
    gb_pred = gb_score >= gb_info["threshold"]
    y = y_test.values

    rng = np.random.default_rng(SEED)
    size = len(y)
    f1_rf, f1_gb = [], []
    pr_rf, pr_gb = [], []
    roc_rf, roc_gb = [], []
    for _ in range(N_ROUNDS):
        index = rng.integers(0, size, size)
        y_sample = y[index]
        f1_rf.append(f1_score(y_sample, rf_pred[index]))
        f1_gb.append(f1_score(y_sample, gb_pred[index]))
        pr_rf.append(average_precision_score(y_sample, rf_score[index]))
        pr_gb.append(average_precision_score(y_sample, gb_score[index]))
        roc_rf.append(roc_auc_score(y_sample, rf_score[index]))
        roc_gb.append(roc_auc_score(y_sample, gb_score[index]))

    rows = []
    for metric, values_rf, values_gb in [
        ("f1", f1_rf, f1_gb),
        ("pr_auc", pr_rf, pr_gb),
        ("roc_auc", roc_rf, roc_gb),
    ]:
        difference = np.array(values_gb) - np.array(values_rf)
        rows.append(summarize(metric, "rf", values_rf))
        rows.append(summarize(metric, "gb", values_gb))
        rows.append(summarize(metric, "gb_minus_rf", difference))
        rows[-1]["share_gb_better"] = (difference > 0).mean()

    table = pd.DataFrame(rows).round(4)
    table.to_csv(RESULT_DIR / "bootstrap_ci.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
