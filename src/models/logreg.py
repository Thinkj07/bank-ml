from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression

from src.config import load_config

SEED = load_config()["random_state"]


def build(C=1.0, penalty="l2", class_weight=None):
    return LogisticRegression(
        C=C,
        penalty=penalty,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )


def build_smote():
    return Pipeline([("smote", SMOTE(random_state=SEED)), ("clf", build())])


def search_space():
    return {
        "C": loguniform(0.001, 100),
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    }


def search_space_smote():
    return {
        "clf__C": loguniform(0.001, 100),
        "clf__penalty": ["l1", "l2"],
    }
