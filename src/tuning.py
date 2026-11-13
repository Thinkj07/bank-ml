from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src.config import load_config
from src.evaluate import evaluate, log_result

SEED = load_config()["random_state"]


def make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune(estimator, param_distributions, X, y, n_iter=20, n_jobs=-1):
    search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=make_cv(),
        n_jobs=n_jobs,
        random_state=SEED,
    )
    search.fit(X, y)
    return search


def best_f1_threshold(y_true, score):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[f1.argmax()])


def oof_scores(estimator, X, y, n_jobs=1):
    proba = cross_val_predict(
        clone(estimator), X, y, cv=make_cv(), method="predict_proba", n_jobs=n_jobs
    )
    return proba[:, 1]


def run_experiment(
    name, estimator, X_train, y_train, X_test, y_test, tune_threshold=True, n_jobs=1, extra=None
):
    model = clone(estimator)
    model.fit(X_train, y_train)

    threshold = None
    if tune_threshold:
        score = oof_scores(estimator, X_train, y_train, n_jobs)
        threshold = best_f1_threshold(y_train, score)

    metrics = evaluate(model, X_test, y_test, threshold)
    if threshold is None:
        metrics["threshold"] = 0.5
    else:
        metrics["threshold"] = threshold
    if extra:
        metrics.update(extra)
    log_result(name, metrics)
    return model, metrics
