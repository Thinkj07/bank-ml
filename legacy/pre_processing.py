"""Bank Marketing data preprocessing pipeline.

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "bank-full.csv"
OUTPUT_PATH = PROJECT_DIR / "cleaned_bank_data.csv"
PLOT_DIR = PROJECT_DIR / "outputs" / "plots"

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", palette="Set2")


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


def main() -> None:
    """Run all preprocessing steps and export the cleaned dataset."""
    data = pd.read_csv(INPUT_PATH, sep=";")

    print_section("1", "Tải dữ liệu và cấu hình môi trường")
    print("[DỮ LIỆU] 5 dòng đầu:")
    display(data.head())
    print("[DỮ LIỆU] 5 dòng cuối:")
    display(data.tail())
    print(f"[DỮ LIỆU] Kích thước: {data.shape[0]:,} dòng x {data.shape[1]} cột")
    print("[DỮ LIỆU] Thông tin tổng quát:")
    data.info()
    print("[DỮ LIỆU] Kiểu dữ liệu từng cột:")
    display(data.dtypes.to_frame("dtype"))

    print_section("2", "Kiểm tra và xử lý giá trị thiếu")
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=np.number).columns.tolist()
    null_count = data.isna().sum()
    unknown_count = data.eq("unknown").sum()
    missing_report = pd.DataFrame(
        {
            "null_count": null_count,
            "unknown_count": unknown_count,
            "total_missing_or_unknown": null_count + unknown_count,
        }
    )
    missing_report["percentage"] = (
        missing_report["total_missing_or_unknown"] / len(data) * 100
    )
    print("[THIẾU] Báo cáo Null/unknown:")
    display(missing_report.sort_values("total_missing_or_unknown", ascending=False))

    plot_report = missing_report[
        missing_report["total_missing_or_unknown"] > 0
    ].sort_values("percentage")
    plt.figure(figsize=(10, 5))
    bars = plt.barh(plot_report.index, plot_report["percentage"], color="#4C78A8")
    plt.title("Tỷ lệ Null hoặc unknown theo cột")
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

    for column in numeric_columns:
        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())
    for column in categorical_columns:
        if data[column].isna().any():
            data[column] = data[column].fillna(data[column].mode().iloc[0])
    print(f"[THIẾU] Tổng số null sau xử lý: {int(data.isna().sum().sum())}")

    print_section("3", "Kiểm tra và xử lý dữ liệu trùng lặp")
    duplicate_count = int(data.duplicated().sum())
    print(f"[TRÙNG LẶP] Số dòng trùng lặp: {duplicate_count:,}")
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"[TRÙNG LẶP] Kích thước sau xử lý: {data.shape[0]:,} dòng x {data.shape[1]} cột")

    print_section("4", "Phân tích và xử lý giá trị ngoại lệ")
    numeric_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]
    capping_columns = ["age", "balance", "day", "duration", "campaign"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, column in zip(axes.flat, numeric_columns):
        sns.boxplot(y=data[column], ax=axis, color="#72B7B2")
        axis.set_title(f"Trước xử lý: {column}")
        axis.set_ylabel(column)
    axes.flat[-1].axis("off")
    fig.suptitle("Boxplot các biến số trước khi cắt ngọn ngoại lệ", fontsize=15)
    plt.tight_layout()
    save_plot("02_outliers_before_capping.png")

    outlier_summary = {}
    for column in capping_columns:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_summary[column] = int(
            ((data[column] < lower) | (data[column] > upper)).sum()
        )
        data[column] = data[column].clip(lower=lower, upper=upper)
    print("[NGOẠI LỆ] Số ngoại lệ trước capping:")
    display(pd.Series(outlier_summary, name="số ngoại lệ trước capping"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, column in zip(axes.flat, numeric_columns):
        sns.boxplot(y=data[column], ax=axis, color="#F2CF5B")
        axis.set_title(f"Sau xử lý: {column}")
        axis.set_ylabel(column)
    axes.flat[-1].axis("off")
    fig.suptitle("Boxplot các biến số sau khi capping theo IQR", fontsize=15)
    plt.tight_layout()
    save_plot("03_outliers_after_capping.png")

    print_section("5", "Mã hóa biến phân loại")
    features_before_encoding = data.shape[1]
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
    data = pd.get_dummies(
        data, columns=["job", "marital", "contact", "poutcome"], dtype="int8"
    )
    features_after_encoding = data.shape[1]
    print(f"[ENCODING] Số cột trước mã hóa: {features_before_encoding}")
    print(f"[ENCODING] Số cột sau mã hóa: {features_after_encoding}")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Trước encoding", "Sau encoding"],
        [features_before_encoding, features_after_encoding],
        color=["#4C78A8", "#F58518"],
    )
    plt.title("So sánh số lượng đặc trưng trước và sau One-Hot Encoding")
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

    print_section("6", "Chuẩn hóa và biến đổi thang đo")
    skewness = data[numeric_columns].skew().sort_values(ascending=False)
    print("[SCALING] Độ lệch của các biến số:")
    display(skewness.to_frame("skewness"))
    scaler = RobustScaler()
    scaled_data = data.copy()
    scaled_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    fig, axes = plt.subplots(len(numeric_columns), 2, figsize=(14, 22))
    for row, column in enumerate(numeric_columns):
        sns.histplot(data[column], kde=True, ax=axes[row, 0], color="#4C78A8")
        axes[row, 0].set_title(f"{column} - trước scaling")
        axes[row, 0].set_xlabel(column)
        axes[row, 0].set_ylabel("Tần số")
        sns.histplot(
            scaled_data[column], kde=True, ax=axes[row, 1], color="#E45756"
        )
        axes[row, 1].set_title(f"{column} - sau RobustScaler")
        axes[row, 1].set_xlabel(column)
        axes[row, 1].set_ylabel("Tần số")
    plt.tight_layout()
    save_plot("05_distributions_before_after_scaling.png")
    data = scaled_data
    print("[SCALING] Thống kê sau RobustScaler:")
    display(data[numeric_columns].describe().T[["mean", "50%", "std", "min", "max"]])

    print_section("7", "Phân tích tương quan và đa cộng tuyến")
    correlation_matrix = data.corr(numeric_only=True)
    target_correlation = correlation_matrix["y"].drop("y").sort_values(
        key=np.abs, ascending=False
    )
    print("[TƯƠNG QUAN] 10 đặc trưng tương quan mạnh nhất với y:")
    display(target_correlation.to_frame("correlation_with_y").head(10))

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        correlation_matrix,
        cmap="vlag",
        center=0,
        linewidths=0.2,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Ma trận tương quan của các đặc trưng sau tiền xử lý")
    plt.xlabel("Đặc trưng")
    plt.ylabel("Đặc trưng")
    plt.tight_layout()
    save_plot("06_correlation_heatmap.png")

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    strong_pairs = upper_triangle.stack().abs().sort_values(ascending=False).to_frame(
        "absolute_correlation"
    )
    print("[ĐA CỘNG TUYẾN] 10 cặp có tương quan tuyệt đối cao nhất:")
    display(strong_pairs.head(10))

    X = data.drop(columns="y")
    y = data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print_section("8", "Tách dữ liệu và xử lý mất cân bằng lớp")
    print("[MẤT CÂN BẰNG] Phân phối y ban đầu:")
    display(y.value_counts(normalize=True).sort_index().mul(100).round(2).rename("percentage"))

    initial_counts = y.value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["No", "Yes"], initial_counts.values, color=["#4C78A8", "#F58518"])
    plt.title("Phân phối biến mục tiêu trước khi xử lý mất cân bằng")
    plt.xlabel("Kết quả đăng ký")
    plt.ylabel("Số lượng mẫu")
    for bar, count in zip(bars, initial_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,} ({count / len(y) * 100:.2f}%)",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    save_plot("07_target_distribution_before_smote.png")

    X_train_resampled, y_train_resampled = SMOTE(
        random_state=RANDOM_STATE
    ).fit_resample(X_train, y_train)
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
    comparison_plot = comparison_percentages.rename(index={0: "No", 1: "Yes"}).plot(
        kind="bar", figsize=(8, 5), color=["#4C78A8", "#54A24B"]
    )
    comparison_plot.set_title("Tỷ lệ y_train trước và sau SMOTE")
    comparison_plot.set_xlabel("Lớp mục tiêu")
    comparison_plot.set_ylabel("Tỷ lệ (%)")
    comparison_plot.set_xticklabels(["No", "Yes"], rotation=0)
    comparison_plot.legend(title="Trạng thái")
    for container in comparison_plot.containers:
        comparison_plot.bar_label(container, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    save_plot("08_train_distribution_before_after_smote.png")

    print(f"[SMOTE] Kích thước X_train: {X_train.shape}")
    print(f"[SMOTE] Kích thước X_train_resampled: {X_train_resampled.shape}")
    print(f"[SMOTE] Kích thước X_test: {X_test.shape}")
    print("[SMOTE] Tỷ lệ y_test giữ nguyên sau khi chỉ SMOTE tập train:")
    display(y_test.value_counts(normalize=True).sort_index().mul(100).round(2).rename("percentage"))

    print_section("9", "Xuất dữ liệu đã chuẩn hóa")
    try:
        data.to_csv(OUTPUT_PATH, index=False)
    except PermissionError:
        fallback_path = OUTPUT_PATH.with_name("cleaned_bank_data_new.csv")
        data.to_csv(fallback_path, index=False)
        print("[XUẤT FILE] Không thể ghi đè cleaned_bank_data.csv. "
              "Hãy đóng file đang mở để dùng đúng tên đầu ra.")
        output_path = fallback_path
    else:
        output_path = OUTPUT_PATH

    print(f"[XUẤT FILE] Đã lưu dữ liệu: {output_path}")
    print(f"[TỔNG KẾT] Kích thước cuối: {data.shape[0]:,} dòng x {data.shape[1]} cột")
    print(f"[TỔNG KẾT] Tổng số giá trị null: {int(data.isna().sum().sum())}")
    print(f"[TỔNG KẾT] Thư mục biểu đồ: {PLOT_DIR}")


if __name__ == "__main__":
    main()
