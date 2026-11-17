"""Huấn luyện, tuning và đánh giá Random Forest (thành viên C).

Chạy từ thư mục gốc của repo:
    python -m scripts.train_rf              # mặc định n_iter=30
    python -m scripts.train_rf --n-iter 10  # chạy nhanh hơn

Đầu ra:
    models/rf_tuned.joblib                     mô hình tuned (đã gắn ngưỡng tối ưu)
    reports/results/results.csv                3 dòng rf_* (ghi bằng log_result)
    reports/results/rf_comparison.csv          bảng so sánh baseline / tuned / tuned+ngưỡng
    reports/results/rf_best_params.json        tham số tốt nhất, F1 CV, ngưỡng
    reports/results/rf_cv_results.csv          toàn bộ kết quả RandomizedSearchCV
    reports/figures/rf_*.png                   feature importance, confusion matrix, ROC/PR
"""

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.config import ROOT, load_config  # noqa: E402
from src.data import load_train_test  # noqa: E402
from src.evaluate import evaluate, log_result  # noqa: E402
from src.models.rf import (  # noqa: E402
    PARAM_DISTRIBUTIONS,
    ThresholdedModel,
    best_f1_threshold,
    build,
)

NAMES = ["rf_baseline", "rf_tuned", "rf_tuned_threshold"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=30, help="số tổ hợp tham số thử")
    parser.add_argument("--cv", type=int, default=5, help="số fold")
    return parser.parse_args()


def drop_old_rf_rows():
    """Xóa các dòng rf_* cũ trong results.csv để chạy lại không bị trùng."""
    path = ROOT / load_config()["paths"]["results"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        old[~old["model"].isin(NAMES)].to_csv(path, index=False)


def main():
    args = parse_args()
    seed = load_config()["random_state"]
    results_dir = ROOT / "reports" / "results"
    figures_dir = ROOT / "reports" / "figures"
    models_dir = ROOT / "models"
    for folder in (results_dir, figures_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # 1. Dữ liệu chung của nhóm (không tự chia lại)
    X_train, X_test, y_train, y_test = load_train_test()
    print(f"train {X_train.shape}, test {X_test.shape}, tỷ lệ yes train {y_train.mean():.4f}")

    # 2. Baseline: Random Forest gần như mặc định, không xử lý mất cân bằng
    baseline = build(n_estimators=100, class_weight=None).fit(X_train, y_train)

    # 3. Tuning bằng RandomizedSearchCV, chọn theo F1 của lớp "yes"
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=build(n_jobs=1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=args.n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    print("Tham số tốt nhất:", search.best_params_)
    print(f"F1 trung bình {args.cv} fold: {search.best_score_:.4f}")

    # 4. Chọn ngưỡng quyết định trên xác suất out-of-fold của train
    oof_proba = cross_val_predict(
        build(**search.best_params_), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold, oof_f1 = best_f1_threshold(y_train, oof_proba)
    print(f"Ngưỡng tối ưu (OOF train): {threshold:.2f}, F1 OOF {oof_f1:.4f}")
    tuned_thr = ThresholdedModel(tuned, threshold)

    # 5. Đánh giá trên test và ghi kết quả chung
    models = dict(zip(NAMES, [baseline, tuned, tuned_thr]))
    drop_old_rf_rows()
    rows = {}
    for name, model in models.items():
        rows[name] = evaluate(model, X_test, y_test)
        log_result(name, rows[name])
    comparison = pd.DataFrame(rows).T.round(4)
    comparison.index.name = "model"
    print("\nSo sánh trên tập test:")
    print(comparison)
    comparison.to_csv(results_dir / "rf_comparison.csv")

    with open(results_dir / "rf_best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": search.best_params_,
                "cv_f1": round(search.best_score_, 4),
                "threshold": threshold,
                "oof_f1_at_threshold": round(oof_f1, 4),
                "n_iter": args.n_iter,
                "cv_folds": args.cv,
            },
            f,
            indent=2,
            default=str,
        )
    cv_table = pd.DataFrame(search.cv_results_)
    keep = [c for c in cv_table.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"
    ]
    cv_table[keep].sort_values("rank_test_score").to_csv(results_dir / "rf_cv_results.csv", index=False)

    # 6. Lưu mô hình
    joblib.dump(tuned_thr, models_dir / "rf_tuned.joblib")

    # 7. Feature importance: MDI và permutation importance (trên test, theo PR-AUC)
    mdi = pd.Series(tuned.feature_importances_, index=X_train.columns)
    perm = permutation_importance(
        tuned, X_test, y_test, scoring="average_precision",
        n_repeats=5, random_state=seed, n_jobs=-1,
    )
    importance = pd.DataFrame(
        {"mdi": mdi, "permutation_mean": perm.importances_mean, "permutation_std": perm.importances_std}
    ).sort_values("permutation_mean", ascending=False)
    importance.round(5).to_csv(results_dir / "rf_feature_importance.csv")
    print("\nTop 15 đặc trưng (permutation, PR-AUC):")
    print(importance.head(15).round(4))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    mdi.sort_values().tail(15).plot(kind="barh", color="#4C78A8", ax=axes[0])
    axes[0].set_title("MDI (impurity) - top 15")
    axes[0].set_xlabel("Importance")
    top_perm = importance.head(15).iloc[::-1]
    axes[1].barh(top_perm.index, top_perm["permutation_mean"], xerr=top_perm["permutation_std"], color="#F58518")
    axes[1].set_title("Permutation importance trên test (giảm PR-AUC) - top 15")
    axes[1].set_xlabel("Mức giảm PR-AUC")
    fig.suptitle("Random Forest tuned - đặc trưng quan trọng")
    fig.tight_layout()
    fig.savefig(figures_dir / "rf_feature_importance.png", dpi=150)
    plt.close(fig)

    # 8. Confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["Baseline", "Tuned (ngưỡng 0.5)", f"Tuned (ngưỡng {threshold:.2f})"]
    for ax, model, title in zip(axes, models.values(), titles):
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(X_test), display_labels=["No", "Yes"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figures_dir / "rf_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 9. ROC và Precision-Recall
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, model in [("Baseline", baseline), ("Tuned", tuned)]:
        proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="Ngẫu nhiên")
    axes[1].set_title("Precision-Recall")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "rf_roc_pr.png", dpi=150)
    plt.close(fig)

    print("\nĐã lưu mô hình, bảng kết quả và hình vào models/ và reports/.")


if __name__ == "__main__":
    main()
