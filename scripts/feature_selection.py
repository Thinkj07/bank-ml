import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from src.config import ROOT
from src.data import load_train_test
from src.models import gb, rf
from src.plots import FIGURE_DIR, plot_importance
from src.tuning import SEED, make_cv, run_experiment

RESULT_DIR = ROOT / "reports" / "results"


def read_json(filename):
    with open(RESULT_DIR / filename, encoding="utf-8") as file:
        return json.load(file)


def rank_features(X, y, gb_params):
    total = pd.Series(0.0, index=X.columns)
    folds = 0
    for train_index, valid_index in make_cv().split(X, y):
        model = gb.build(**gb_params)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        result = permutation_importance(
            model,
            X.iloc[valid_index],
            y.iloc[valid_index],
            scoring="average_precision",
            n_repeats=5,
            random_state=SEED,
        )
        total = total + result.importances_mean
        folds = folds + 1
    return (total / folds).sort_values(ascending=False)


def cv_by_size(X, y, ranking, sizes, gb_params):
    rows = []
    for size in sizes:
        columns = list(ranking.index[:size])
        scores = cross_val_score(
            gb.build(**gb_params),
            X[columns],
            y,
            scoring="average_precision",
            cv=make_cv(),
            n_jobs=2,
        )
        rows.append({"n_features": size, "cv_pr_auc": scores.mean(), "cv_std": scores.std()})
        print(size, round(scores.mean(), 4), round(scores.std(), 4))
    return pd.DataFrame(rows)


def main():
    X_train, X_test, y_train, y_test = load_train_test()
    gb_params = read_json("gb_best_params.json")["best_params"]
    rf_params = read_json("rf_best_params.json")["best_params"]

    ranking = rank_features(X_train, y_train, gb_params)
    ranking.to_csv(RESULT_DIR / "fs_ranking.csv", header=["permutation_pr_auc_drop"])
    print(ranking.round(4))

    total = X_train.shape[1]
    sizes = [5, 8, 10, 12, 15, 20, 25, total]
    curve = cv_by_size(X_train, y_train, ranking, sizes, gb_params)
    curve.to_csv(RESULT_DIR / "fs_cv_curve.csv", index=False)

    best_score = curve["cv_pr_auc"].max()
    good = curve[curve["cv_pr_auc"] >= best_score - 0.002]
    chosen_size = int(good["n_features"].min())
    chosen = list(ranking.index[:chosen_size])
    print("số đặc trưng được chọn:", chosen_size)
    print(chosen)

    with open(RESULT_DIR / "fs_selected.json", "w", encoding="utf-8") as file:
        json.dump({"n_features": chosen_size, "features": chosen}, file, indent=2)

    gb_model, gb_metrics = run_experiment(
        "gb_fs",
        gb.build(**gb_params),
        X_train[chosen],
        y_train,
        X_test[chosen],
        y_test,
        tune_threshold=True,
        extra={"n_features": chosen_size},
    )
    print("gb_fs", gb_metrics)

    rf_model, rf_metrics = run_experiment(
        "rf_fs",
        rf.build(**rf_params),
        X_train[chosen],
        y_train,
        X_test[chosen],
        y_test,
        tune_threshold=True,
        n_jobs=2,
        extra={"n_features": chosen_size},
    )
    print("rf_fs", rf_metrics)

    plot_importance(
        ranking,
        "Permutation importance trên train (giảm PR-AUC)",
        "fs_importance.png",
        top=15,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(curve["n_features"], curve["cv_pr_auc"], yerr=curve["cv_std"], marker="o")
    ax.axvline(chosen_size, linestyle="--", color="gray")
    ax.set_xlabel("Số đặc trưng giữ lại")
    ax.set_ylabel("PR-AUC cross-validation")
    ax.set_title("Chọn số đặc trưng")
    fig.savefig(FIGURE_DIR / "fs_cv_curve.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
