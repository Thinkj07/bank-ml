import pandas as pd
from sklearn.preprocessing import RobustScaler

BINARY_COLUMNS = ["default", "housing", "loan"]
ONE_HOT_COLUMNS = ["job", "marital", "contact", "poutcome"]
EDUCATION_MAP = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def load_raw(path):
    return pd.read_csv(path, sep=";")


def encode(data, target="y", drop_duration=True):
    data = data.drop_duplicates().reset_index(drop=True)
    if drop_duration:
        data = data.drop(columns="duration")
    for column in BINARY_COLUMNS + [target]:
        data[column] = data[column].map({"no": 0, "yes": 1}).astype("int8")
    data["education"] = data["education"].map(EDUCATION_MAP).astype("int8")
    data["month"] = data["month"].map(MONTH_MAP).astype("int8")
    return pd.get_dummies(data, columns=ONE_HOT_COLUMNS, dtype="int8")


class Preprocessor:
    def __init__(self, numeric_columns, capping_columns):
        self.numeric_columns = list(numeric_columns)
        self.capping_columns = list(capping_columns)
        self.bounds_ = {}
        self.scaler_ = RobustScaler()

    def _clip(self, data):
        data = data.copy()
        for column, (lower, upper) in self.bounds_.items():
            data[column] = data[column].clip(lower=lower, upper=upper)
        return data

    def fit(self, data):
        self.bounds_ = {}
        for column in self.capping_columns:
            q1, q3 = data[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        self.scaler_.fit(self._clip(data)[self.numeric_columns])
        return self

    def transform(self, data):
        data = self._clip(data)
        data[self.numeric_columns] = self.scaler_.transform(data[self.numeric_columns])
        return data
