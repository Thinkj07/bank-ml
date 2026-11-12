from sklearn.linear_model import LogisticRegression


def build(**params):
    return LogisticRegression(max_iter=1000, class_weight="balanced", **params)
