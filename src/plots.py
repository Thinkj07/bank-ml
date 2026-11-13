import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from src.config import ROOT

FIGURE_DIR = ROOT / "reports" / "figures"


def _save(fig, filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")


def plot_confusion(model, X, y, threshold, title, filename):
    pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["No", "Yes"], ax=ax, colorbar=False
    )
    ax.set_title(title)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    _save(fig, filename)
    return fig


def plot_pr_curves(models, X, y, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X, y, name=name, ax=ax)
    ax.axhline(y.mean(), linestyle="--", color="gray", label="Ngẫu nhiên")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)
    return fig


def plot_importance(values, title, filename, top=15):
    order = values.abs().sort_values(ascending=False).index[:top]
    values = values[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    values[::-1].plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title(title)
    _save(fig, filename)
    return fig
