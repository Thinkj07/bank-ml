import json

from src.config import ROOT
from src.data import load_train_test
from src.models import gb
from src.plots import plot_confusion, plot_pr_curves
from src.tuning import run_experiment, tune


def to_plain(params):
    plain = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        plain[key] = value
    return plain


def main():
    X_train, X_test, y_train, y_test = load_train_test()

    baseline, baseline_metrics = run_experiment(
        "gb_baseline", gb.build(), X_train, y_train, X_test, y_test, tune_threshold=False
    )
    print("gb_baseline", baseline_metrics)

    baseline_thr, baseline_thr_metrics = run_experiment(
        "gb_baseline_thr", gb.build(), X_train, y_train, X_test, y_test, tune_threshold=True
    )
    print("gb_baseline_thr", baseline_thr_metrics)

    search = tune(gb.build(), gb.search_space(), X_train, y_train, n_iter=30, n_jobs=2)
    best_params = to_plain(search.best_params_)
    print("best params", best_params, "cv pr_auc", search.best_score_)

    tuned, tuned_metrics = run_experiment(
        "gb_tuned",
        gb.build(**best_params),
        X_train,
        y_train,
        X_test,
        y_test,
        tune_threshold=True,
        extra={"cv_pr_auc": search.best_score_},
    )
    print("gb_tuned", tuned_metrics)

    saved = {
        "best_params": best_params,
        "cv_pr_auc": round(float(search.best_score_), 4),
        "threshold": tuned_metrics["threshold"],
    }
    with open(ROOT / "reports" / "results" / "gb_best_params.json", "w", encoding="utf-8") as file:
        json.dump(saved, file, indent=2)

    plot_confusion(
        tuned,
        X_test,
        y_test,
        tuned_metrics["threshold"],
        "Gradient Boosting tuned",
        "gb_confusion.png",
    )
    plot_pr_curves(
        {"GB baseline": baseline, "GB tuned": tuned},
        X_test,
        y_test,
        "Đường Precision-Recall của Gradient Boosting",
        "gb_pr_curves.png",
    )


if __name__ == "__main__":
    main()
