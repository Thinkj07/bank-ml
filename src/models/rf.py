import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

DEFAULTS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 501),
    "max_depth": [None, 8, 12, 16, 20, 25, 30],
    "min_samples_split": randint(2, 21),
    "min_samples_leaf": randint(1, 11),
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced", "balanced_subsample"],
}


def build(**params):
    return RandomForestClassifier(**{**DEFAULTS, **params})


class ThresholdedModel:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def best_f1_threshold(y_true, proba):
    y_true = np.asarray(y_true)
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = []
    for t in thresholds:
        pred = proba >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum(~pred & (y_true == 1))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])
