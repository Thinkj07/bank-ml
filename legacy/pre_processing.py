"""Bank Marketing data preprocessing pipeline without Data Leakage.

Run from the project directory:
    python pre_processing.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESAMPLED_DIR = PROJECT_ROOT / "data" / "resampled"
PLOT_DIR = PROJECT_ROOT / "reports" / "figures"

TRAIN_PROCESSED_PATH = PROCESSED_DIR / "train_processed.csv"
TRAIN_RESAMPLED_PATH = RESAMPLED_DIR / "train_resampled.csv"
TEST_PROCESSED_PATH = PROCESSED_DIR / "test_processed.csv"

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", palette="Set2")


class IQRCapper(BaseEstimator, TransformerMixin):
    """Chỉ học ngưỡng Q1, Q3 từ tập train và áp dụng clip cho cả train/test."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "IQRCapper":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=str)
        X_arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X_arr, 25, axis=0)
        q3 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=str)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return None


def print_section(step: str, title: str) -> None:
    """Print a consistently formatted pipeline section heading."""
    print(f"\n{'=' * 72}\nBƯỚC {step}: {title}\n{'=' * 72}")


def save_plot(filename: str) -> None:
    """Save the current figure in the dedicated plots directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[BIỂU ĐỒ] Đã lưu: {output_path}")
    plt.close()


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Save DataFrame to CSV, handling file lock gracefully."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new.csv")
        df.to_csv(fallback, index=False)
        print(
            f"[CẢNH BÁO] Không thể ghi vào {path.name} (file có thể đang mở). "
            f"Đã lưu vào {fallback.name}"
        )
        return fallback


def main() -> None:
    """Run all preprocessing steps without Data Leakage."""
    # =========================================================================
    # BƯỚC 1: Tải dữ liệu & làm sạch tĩnh (Trước khi chia train/test)
    # =========================================================================
    print_section("1", "Tải dữ liệu & làm sạch tĩnh (Trước khi chia train/test)")
    data = pd.read_csv(INPUT_PATH, sep=";")

    print("[DỮ LIỆU] 5 dòng đầu:")
    display(data.head())
    print("[DỮ LIỆU] 5 dòng cuối:")
    display(data.tail())
    print(f"[DỮ LIỆU] Kích thước ban đầu: {data.shape[0]:,} dòng x {data.shape[1]} cột")
    print("[DỮ LIỆU] Thông tin tổng quát:")
    data.info()
    print("[DỮ LIỆU] Kiểu dữ liệu từng cột:")
    display(data.dtypes.to_frame("dtype"))

    # Xóa dòng trùng lặp
    duplicate_count = int(data.duplicated().sum())
    print(f"[TRÙNG LẶP] Số dòng trùng lặp: {duplicate_count:,}")
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"[TRÙNG LẶP] Kích thước sau drop_duplicates: {data.shape[0]:,} dòng x {data.shape[1]} cột")

    # Loại bỏ cột duration (không còn dùng duration nữa)
    if "duration" in data.columns:
        data = data.drop(columns=["duration"])
        print(f"[LOẠI BỎ] Đã loại bỏ cột duration. Kích thước hiện tại: {data.shape[0]:,} dòng x {data.shape[1]} cột")

    # Áp dụng các phép ánh xạ cố định (không phụ thuộc phân phối dữ liệu)
    for column in ["default", "housing", "loan", "y"]:
        data[column] = data[column].map({"no": 0, "yes": 1}).astype("int8")

    data["education"] = data["education"].map(
        {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    ).astype("int8")

    data["month"] = data["month"].map(
        {
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
    ).astype("int8")
    print("[ÁNH XẠ] Đã map tĩnh các cột: default, housing, loan, y, education, month")

    # =========================================================================
    # BƯỚC 2: Chia tập dữ liệu (train_test_split sớm)
    # =========================================================================
    print_section("2", "Chia tập dữ liệu (train_test_split sớm để chống rò rỉ dữ liệu)")
    X = data.drop(columns="y")
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"[CHIA TẬP] X_train: {X_train.shape[0]:,} dòng x {X_train.shape[1]} cột")
    print(f"[CHIA TẬP] X_test:  {X_test.shape[0]:,} dòng x {X_test.shape[1]} cột")
    print(f"[CHIA TẬP] y_train: {y_train.shape[0]:,} nhãn")
    print(f"[CHIA TẬP] y_test:  {y_test.shape[0]:,} nhãn")

    # =========================================================================
    # BƯỚC 3: EDA & Trực quan hóa (Chỉ thực hiện trên tập Train)
    # =========================================================================
    print_section("3", "EDA & Trực quan hóa (Chỉ thực hiện trên tập Train)")
    null_count = X_train.isna().sum()
    unknown_count = X_train.eq("unknown").sum()
    if "education" in X_train.columns:
        unknown_count["education"] = (X_train["education"] == 0).sum()

    missing_report = pd.DataFrame(
        {
            "null_count": null_count,
            "unknown_count": unknown_count,
            "total_missing_or_unknown": null_count + unknown_count,
        }
    )
    missing_report["percentage"] = (
        missing_report["total_missing_or_unknown"] / len(X_train) * 100
    )
    print("[THIẾU] Báo cáo Null/unknown trên tập Train:")
    display(missing_report.sort_values("total_missing_or_unknown", ascending=False))

    plot_report = missing_report[
        missing_report["total_missing_or_unknown"] > 0
    ].sort_values("percentage")
    plt.figure(figsize=(10, 5))
    bars = plt.barh(plot_report.index, plot_report["percentage"], color="#4C78A8")
    plt.title("Tỷ lệ Null hoặc unknown theo cột (Chỉ tập Train)")
    plt.xlabel("Tỷ lệ (%)")
    plt.ylabel("Cột")
    for bar, value in zip(bars, plot_report["percentage"]):
        plt.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )
    plt.tight_layout()
    save_plot("01_missing_unknown_rate.png")

    numeric_columns = [
        "age",
        "balance",
        "day",
        "campaign",
        "pdays",
        "previous",
    ]
    capping_columns = ["age", "balance", "day", "campaign"]
    other_numeric_columns = ["pdays", "previous"]
    categorical_columns = ["job", "marital", "contact", "poutcome"]

    # Boxplot các biến số trên tập Train trước khi capping
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, column in zip(axes.flat, numeric_columns):
        sns.boxplot(y=X_train[column], ax=axis, color="#72B7B2")
        axis.set_title(f"Trước xử lý: {column}")
        axis.set_ylabel(column)
    fig.suptitle("Boxplot các biến số trước khi cắt ngọn ngoại lệ (Chỉ tập Train)", fontsize=15)
    plt.tight_layout()
    save_plot("02_outliers_before_capping.png")

    # =========================================================================
    # BƯỚC 4: Xây dựng ColumnTransformer & Pipeline (Fit trên Train, Transform trên Test)
    # =========================================================================
    print_section(
        "4", "Xây dựng ColumnTransformer & Pipeline (Fit trên Train, Transform trên Test)"
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num_capped",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("capper", IQRCapper(factor=1.5)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                capping_columns,
            ),
            (
                "num_other",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                other_numeric_columns,
            ),
            (
                "cat_ohe",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.int8,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    preprocessor.set_output(transform="pandas")

    # Fit và transform trên tập Train, CHỈ transform trên tập Test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"[TRANSFORM] Kích thước X_train sau pipeline: {X_train_processed.shape}")
    print(f"[TRANSFORM] Kích thước X_test sau pipeline:  {X_test_processed.shape}")

    # Trích xuất thông số capper đã học từ tập Train để báo cáo & trực quan hóa
    capper = preprocessor.named_transformers_["num_capped"].named_steps["capper"]
    outlier_summary = {}
    for idx, column in enumerate(capping_columns):
        lower = capper.lower_[idx]
        upper = capper.upper_[idx]
        outlier_summary[column] = int(
            ((X_train[column] < lower) | (X_train[column] > upper)).sum()
        )
    print("[NGOẠI LỆ] Số ngoại lệ trên tập Train trước capping (ngưỡng học từ Train):")
    display(pd.Series(outlier_summary, name="số ngoại lệ trước capping"))

    # Boxplot các biến số sau khi capping theo IQR (chỉ tập Train)
    X_train_capped = X_train[numeric_columns].copy()
    X_train_capped[capping_columns] = np.clip(
        X_train[capping_columns].to_numpy(dtype=float), capper.lower_, capper.upper_
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, column in zip(axes.flat, numeric_columns):
        sns.boxplot(y=X_train_capped[column], ax=axis, color="#F2CF5B")
        axis.set_title(f"Sau xử lý: {column}")
        axis.set_ylabel(column)
    fig.suptitle("Boxplot các biến số sau khi capping theo IQR (Chỉ tập Train)", fontsize=15)
    plt.tight_layout()
    save_plot("03_outliers_after_capping.png")

    # Trực quan hóa số lượng đặc trưng trước & sau One-Hot Encoding
    features_before_encoding = X_train.shape[1]
    features_after_encoding = X_train_processed.shape[1]
    print(f"[ENCODING] Số cột trước mã hóa: {features_before_encoding}")
    print(f"[ENCODING] Số cột sau mã hóa:   {features_after_encoding}")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Trước encoding", "Sau encoding"],
        [features_before_encoding, features_after_encoding],
        color=["#4C78A8", "#F58518"],
    )
    plt.title("So sánh số lượng đặc trưng trước và sau One-Hot Encoding (Chỉ tập Train)")
    plt.ylabel("Số lượng cột")
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            int(bar.get_height()),
            ha="center",
        )
    plt.tight_layout()
    save_plot("04_features_before_after_encoding.png")

    # Trực quan hóa phân phối trước & sau RobustScaler trên tập Train
    skewness = X_train[numeric_columns].skew().sort_values(ascending=False)
    print("[SCALING] Độ lệch của các biến số (Tập Train):")
    display(skewness.to_frame("skewness"))

    fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(14, 18))
    for row, column in enumerate(numeric_columns):
        sns.histplot(X_train[column], kde=True, ax=axes[row, 0], color="#4C78A8")
        axes[row, 0].set_title(f"{column} - trước scaling")
        axes[row, 0].set_xlabel(column)
        axes[row, 0].set_ylabel("Tần số")
        sns.histplot(
            X_train_processed[column], kde=True, ax=axes[row, 1], color="#E45756"
        )
        axes[row, 1].set_title(f"{column} - sau RobustScaler")
        axes[row, 1].set_xlabel(column)
        axes[row, 1].set_ylabel("Tần số")
    plt.tight_layout()
    save_plot("05_distributions_before_after_scaling.png")
    print("[SCALING] Thống kê sau RobustScaler (Tập Train):")
    display(
        X_train_processed[numeric_columns].describe().T[
            ["mean", "50%", "std", "min", "max"]
        ]
    )

    # Phân tích tương quan & đa cộng tuyến (chỉ trên tập Train đã xử lý kết hợp y_train)
    train_for_corr = pd.concat([X_train_processed, y_train], axis=1)
    correlation_matrix = train_for_corr.corr(numeric_only=True)
    target_correlation = correlation_matrix["y"].drop("y").sort_values(
        key=np.abs, ascending=False
    )
    print("[TƯƠNG QUAN] 10 đặc trưng tương quan mạnh nhất với y (Tập Train):")
    display(target_correlation.to_frame("correlation_with_y").head(10))

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        correlation_matrix,
        cmap="vlag",
        center=0,
        linewidths=0.2,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Ma trận tương quan của các đặc trưng sau tiền xử lý (Chỉ tập Train)")
    plt.xlabel("Đặc trưng")
    plt.ylabel("Đặc trưng")
    plt.tight_layout()
    save_plot("06_correlation_heatmap.png")

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    strong_pairs = (
        upper_triangle.stack()
        .abs()
        .sort_values(ascending=False)
        .to_frame("absolute_correlation")
    )
    print("[ĐA CỘNG TUYẾN] 10 cặp có tương quan tuyệt đối cao nhất (Tập Train):")
    display(strong_pairs.head(10))

    # =========================================================================
    # BƯỚC 5: Xử lý mất cân bằng lớp (SMOTE duy nhất trên tập Train)
    # =========================================================================
    print_section("5", "Xử lý mất cân bằng lớp (SMOTE duy nhất trên tập Train)")
    print("[MẤT CÂN BẰNG] Phân phối y_train ban đầu:")
    display(
        y_train.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .rename("percentage")
    )

    initial_counts = y_train.value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["No", "Yes"], initial_counts.values, color=["#4C78A8", "#F58518"])
    plt.title("Phân phối biến mục tiêu y_train trước khi xử lý mất cân bằng")
    plt.xlabel("Kết quả đăng ký")
    plt.ylabel("Số lượng mẫu")
    for bar, count in zip(bars, initial_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,} ({count / len(y_train) * 100:.2f}%)",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    save_plot("07_target_distribution_before_smote.png")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed, y_train
    )
    print("[SMOTE] Phân phối y_train trước SMOTE:")
    print(y_train.value_counts().sort_index())
    print("[SMOTE] Phân phối y_train_resampled sau SMOTE:")
    print(y_train_resampled.value_counts().sort_index())

    comparison_counts = pd.DataFrame(
        {
            "Trước SMOTE": y_train.value_counts().sort_index(),
            "Sau SMOTE": y_train_resampled.value_counts().sort_index(),
        },
        index=[0, 1],
    )
    comparison_percentages = comparison_counts.div(
        comparison_counts.sum(axis=0), axis=1
    ).mul(100)
    comparison_plot = comparison_percentages.rename(
        index={0: "No", 1: "Yes"}
    ).plot(kind="bar", figsize=(8, 5), color=["#4C78A8", "#54A24B"])
    comparison_plot.set_title("Tỷ lệ y_train trước và sau SMOTE")
    comparison_plot.set_xlabel("Lớp mục tiêu")
    comparison_plot.set_ylabel("Tỷ lệ (%)")
    comparison_plot.set_xticklabels(["No", "Yes"], rotation=0)
    comparison_plot.legend(title="Trạng thái")
    for container in comparison_plot.containers:
        comparison_plot.bar_label(container, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    save_plot("08_train_distribution_before_after_smote.png")

    print(f"[SMOTE] Kích thước X_train_processed: {X_train_processed.shape}")
    print(f"[SMOTE] Kích thước X_train_resampled: {X_train_resampled.shape}")
    print(f"[SMOTE] Kích thước X_test_processed:  {X_test_processed.shape}")
    print("[SMOTE] Tỷ lệ y_test giữ nguyên (không qua SMOTE để đảm bảo khách quan):")
    display(
        y_test.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .rename("percentage")
    )

    # =========================================================================
    # BƯỚC 6: Xuất dữ liệu đầu ra
    # =========================================================================
    print_section("6", "Xuất dữ liệu sạch & chuẩn hóa tách biệt train/test")
    train_processed = pd.concat([X_train_processed, y_train], axis=1)
    train_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    test_processed = pd.concat([X_test_processed, y_test], axis=1)

    path_train = safe_to_csv(train_processed, TRAIN_PROCESSED_PATH)
    path_resampled = safe_to_csv(train_resampled, TRAIN_RESAMPLED_PATH)
    path_test = safe_to_csv(test_processed, TEST_PROCESSED_PATH)

    print(f"[XUẤT FILE] Train đã xử lý:  {path_train} ({train_processed.shape[0]:,} dòng x {train_processed.shape[1]} cột)")
    print(f"[XUẤT FILE] Train sau SMOTE:   {path_resampled} ({train_resampled.shape[0]:,} dòng x {train_resampled.shape[1]} cột)")
    print(f"[XUẤT FILE] Test đã xử lý:   {path_test} ({test_processed.shape[0]:,} dòng x {test_processed.shape[1]} cột)")
    print(f"[TỔNG KẾT] Tổng số null tập train: {int(train_processed.isna().sum().sum())}")
    print(f"[TỔNG KẾT] Tổng số null tập test:  {int(test_processed.isna().sum().sum())}")
    print(f"[TỔNG KẾT] Thư mục biểu đồ: {PLOT_DIR}")


if __name__ == "__main__":
    main()
