import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone

from src.config import ROOT
from src.data import load_train_test
from src.evaluate import evaluate
from src.models import rf
from src.tuning import SEED, best_f1_threshold, oof_scores

RESULT_DIR = ROOT / "reports" / "results"
TOP_FRACTION = 0.10


def run(name, estimator, X_train, y_train, X_test, y_test):
    score_train = oof_scores(estimator, X_train, y_train, n_jobs=2)
    threshold = best_f1_threshold(y_train, score_train)

    model = clone(estimator)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test, threshold)
    metrics["threshold"] = threshold

    score_test = model.predict_proba(X_test)[:, 1]
    calls = int(round(len(y_test) * TOP_FRACTION))
    best_index = np.argsort(-score_test)[:calls]
    metrics["top10_calls"] = calls
    metrics["top10_hits"] = int(y_test.values[best_index].sum())

    print(name, {key: round(float(value), 4) for key, value in metrics.items()})
    return metrics


def main():
    X_train, X_test, y_train, y_test = load_train_test()

    with open(RESULT_DIR / "rf_best_params.json", encoding="utf-8") as file:
        best_params = json.load(file)["best_params"]

    no_weight_params = dict(best_params)
    no_weight_params["class_weight"] = None

    smote_pipeline = Pipeline(
        [
            ("smote", SMOTE(random_state=SEED)),
            ("rf", rf.build(**no_weight_params)),
        ]
    )

    setups = {
        "rf_class_weight": rf.build(**best_params),
        "rf_no_class_weight": rf.build(**no_weight_params),
        "rf_smote": smote_pipeline,
    }

    rows = {}
    for name, estimator in setups.items():
        rows[name] = run(name, estimator, X_train, y_train, X_test, y_test)

    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table["top10_calls"] = table["top10_calls"].astype(int)
    table["top10_hits"] = table["top10_hits"].astype(int)
    table.round(4).to_csv(RESULT_DIR / "smote_rf.csv")


if __name__ == "__main__":
    main()
