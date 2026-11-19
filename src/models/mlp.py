from scipy.stats import loguniform
from sklearn.neural_network import MLPClassifier

from src.config import load_config

SEED = load_config()["random_state"]


def build(hidden_layer_sizes=(64,), alpha=0.0001, learning_rate_init=0.001):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=SEED,
    )


def search_space():
    return {
        "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
        "alpha": loguniform(0.00001, 0.1),
        "learning_rate_init": loguniform(0.0001, 0.01),
    }
