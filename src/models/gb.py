from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import load_config

SEED = load_config()["random_state"]


def build(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
):
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=SEED,
    )


def search_space():
    return {
        "learning_rate": loguniform(0.01, 0.2),
        "max_iter": randint(100, 400),
        "max_leaf_nodes": randint(8, 40),
        "max_depth": [None, 4, 6, 8],
        "min_samples_leaf": randint(20, 200),
        "l2_regularization": loguniform(0.01, 10),
    }
