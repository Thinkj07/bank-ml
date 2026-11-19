import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, roc_auc_score

from src.config import ROOT
from src.data import load_train_test
from src.evaluate import evaluate
from src.models import gb, logreg, mlp, rf
from src.plots import FIGURE_DIR
from src.tuning import best_f1_threshold, oof_scores

RESULT_DIR = ROOT / "reports" / "results"
FRACTIONS = [0.05, 0.10, 0.20, 0.30]


def read_json(filename):
    with open(RESULT_DIR / filename, encoding="utf-8") as file:
        return json.load(file)


def top_k_rows(name, score, y):
    order = np.argsort(-score)
    y_sorted = np.asarray(y)[order]
    total_positive = y_sorted.sum()
    base_rate = y_sorted.mean()
    rows = []
    for fraction in FRACTIONS:
        calls = int(round(len(y_sorted) * fraction))
        hits = y_sorted[:calls].sum()
        precision = hits / calls
        rows.append(
            {
                "model": name,
                "top": f"{int(fraction * 100)}%",
                "calls": calls,
                "hits": int(hits),
                "precision": precision,
                "recall": hits / total_positive,
                "lift": precision / base_rate,
            }
        )
    return rows


def main():
    X_train, X_test, y_train, y_test = load_train_test()

    logreg_params = read_json("logreg_best_params.json")
    rf_info = read_json("rf_best_params.json")
    gb_info = read_json("gb_best_params.json")
    mlp_info = read_json("mlp_best_params.json")
    selected = read_json("fs_selected.json")["features"]

    mlp_params = dict(mlp_info["best_params"])
    mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])

    logreg_model = logreg.build(**logreg_params)
    logreg_threshold = best_f1_threshold(y_train, oof_scores(logreg_model, X_train, y_train))

    results_table = pd.read_csv(RESULT_DIR / "results.csv").set_index("model")
    gb_fs_threshold = results_table.loc["gb_fs", "threshold"]

    setups = {
        "logreg_tuned": (logreg_model, list(X_train.columns), logreg_threshold),
        "mlp_tuned": (mlp.build(**mlp_params), list(X_train.columns), mlp_info["threshold"]),
        "rf_tuned": (rf.build(**rf_info["best_params"]), list(X_train.columns), rf_info["threshold"]),
        "gb_tuned": (gb.build(**gb_info["best_params"]), list(X_train.columns), gb_info["threshold"]),
        "gb_fs": (gb.build(**gb_info["best_params"]), selected, gb_fs_threshold),
    }

    scores = {}
    metric_rows = {}
    for name, (model, columns, threshold) in setups.items():
        model.fit(X_train[columns], y_train)
        metrics = evaluate(model, X_test[columns], y_test, threshold)
        metrics["threshold"] = threshold
        metric_rows[name] = metrics
        scores[name] = model.predict_proba(X_test[columns])[:, 1]
        print(name, {key: round(value, 4) for key, value in metrics.items()})

    blend = 0.5 * rankdata(scores["rf_tuned"]) + 0.5 * rankdata(scores["gb_tuned"])
    scores["blend_rf_gb"] = blend / len(blend)
    blend_metrics = {
        "pr_auc": average_precision_score(y_test, blend),
        "roc_auc": roc_auc_score(y_test, blend),
    }
    print("blend_rf_gb", {key: round(value, 4) for key, value in blend_metrics.items()})

    comparison = pd.DataFrame(metric_rows).T
    comparison.loc["blend_rf_gb", "pr_auc"] = blend_metrics["pr_auc"]
    comparison.loc["blend_rf_gb", "roc_auc"] = blend_metrics["roc_auc"]
    comparison.index.name = "model"
    comparison.round(4).to_csv(RESULT_DIR / "comparison.csv")

    rows = []
    for name, score in scores.items():
        rows.extend(top_k_rows(name, score, y_test))
    topk = pd.DataFrame(rows)
    topk.round(4).to_csv(RESULT_DIR / "topk.csv", index=False)
    print(topk.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["logreg_tuned", "mlp_tuned", "rf_tuned", "gb_tuned"]:
        PrecisionRecallDisplay.from_predictions(y_test, scores[name], name=name, ax=ax)
    ax.axhline(y_test.mean(), linestyle="--", color="gray", label="Ngẫu nhiên")
    ax.set_title("So sánh đường Precision-Recall trên tập test")
    ax.legend()
    fig.savefig(FIGURE_DIR / "compare_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in ["logreg_tuned", "mlp_tuned", "rf_tuned", "gb_tuned"]:
        order = np.argsort(-scores[name])
        gained = np.cumsum(np.asarray(y_test)[order]) / np.sum(y_test)
        fraction = np.arange(1, len(gained) + 1) / len(gained)
        ax.plot(fraction, gained, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Gọi ngẫu nhiên")
    ax.set_xlabel("Tỷ lệ khách được gọi (điểm cao nhất trước)")
    ax.set_ylabel("Tỷ lệ người đồng ý bắt được (recall)")
    ax.set_title("Đường lợi ích tích lũy trên tập test")
    ax.legend()
    fig.savefig(FIGURE_DIR / "compare_gains.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    names = list(metric_rows.keys())
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - 0.2, comparison.loc[names, "f1"], width=0.4, label="F1")
    ax.bar(positions + 0.2, comparison.loc[names, "pr_auc"], width=0.4, label="PR-AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20)
    ax.set_title("F1 và PR-AUC trên tập test")
    ax.legend()
    fig.savefig(FIGURE_DIR / "compare_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
