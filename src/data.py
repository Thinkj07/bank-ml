import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import ROOT, load_config


def _read(name):
    config = load_config()
    return pd.read_csv(ROOT / config["paths"]["processed"] / f"{name}.csv")


def _split_target(data):
    target = load_config()["target"]
    return data.drop(columns=target), data[target]


def load_train():
    return _split_target(_read("train"))


def load_test():
    return _split_target(_read("test"))


def load_train_test():
    X_train, y_train = load_train()
    X_test, y_test = load_test()
    return X_train, X_test, y_train, y_test


def smote_resample(X, y):
    random_state = load_config()["random_state"]
    return SMOTE(random_state=random_state).fit_resample(X, y)
