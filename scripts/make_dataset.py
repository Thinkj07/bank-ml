import hashlib

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config
from src.preprocessing import Preprocessor, encode, load_raw


def checksum(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    config = load_config()
    target = config["target"]
    data = encode(
        load_raw(ROOT / config["paths"]["raw"]),
        target=target,
        drop_duration=config["drop_duration"],
    )
    X = data.drop(columns=target)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    numeric = [c for c in config["numeric_columns"] if c in X.columns]
    capping = [c for c in config["capping_columns"] if c in X.columns]
    preprocessor = Preprocessor(numeric, capping).fit(X_train)

    out_dir = ROOT / config["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.concat([preprocessor.transform(X_train), y_train], axis=1)
    test = pd.concat([preprocessor.transform(X_test), y_test], axis=1)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    print(f"drop_duration={config['drop_duration']}")
    print(f"train: {train.shape}, positive rate {y_train.mean():.4f}")
    print(f"test: {test.shape}, positive rate {y_test.mean():.4f}")
    print(f"train.csv md5 {checksum(out_dir / 'train.csv')}")
    print(f"test.csv md5 {checksum(out_dir / 'test.csv')}")


if __name__ == "__main__":
    main()
