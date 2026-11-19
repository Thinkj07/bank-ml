import json

from src.config import ROOT
from src.data import load_train_test
from src.models import mlp
from src.plots import plot_confusion, plot_pr_curves
from src.tuning import run_experiment, tune


def to_plain(params):
    plain = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, tuple):
            value = list(value)
        plain[key] = value
    return plain


def main():
    X_train, X_test, y_train, y_test = load_train_test()

    baseline, baseline_metrics = run_experiment(
        "mlp_baseline", mlp.build(), X_train, y_train, X_test, y_test, tune_threshold=False
    )
    print("mlp_baseline", baseline_metrics)

    search = tune(mlp.build(), mlp.search_space(), X_train, y_train, n_iter=15, n_jobs=2)
    best_params = to_plain(search.best_params_)
    print("best params", best_params, "cv pr_auc", search.best_score_)

    tuned_params = dict(best_params)
    tuned_params["hidden_layer_sizes"] = tuple(tuned_params["hidden_layer_sizes"])
    tuned, tuned_metrics = run_experiment(
        "mlp_tuned",
        mlp.build(**tuned_params),
        X_train,
        y_train,
        X_test,
        y_test,
        tune_threshold=True,
        extra={"cv_pr_auc": search.best_score_},
    )
    print("mlp_tuned", tuned_metrics)

    saved = {
        "best_params": best_params,
        "cv_pr_auc": round(float(search.best_score_), 4),
        "threshold": tuned_metrics["threshold"],
    }
    with open(ROOT / "reports" / "results" / "mlp_best_params.json", "w", encoding="utf-8") as file:
        json.dump(saved, file, indent=2)

    plot_confusion(
        tuned,
        X_test,
        y_test,
        tuned_metrics["threshold"],
        "MLP tuned",
        "mlp_confusion.png",
    )
    plot_pr_curves(
        {"MLP baseline": baseline, "MLP tuned": tuned},
        X_test,
        y_test,
        "Đường Precision-Recall của MLP",
        "mlp_pr_curves.png",
    )


if __name__ == "__main__":
    main()
