from sklearn.ensemble import RandomForestClassifier


def build(**params):
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42, **params
    )
