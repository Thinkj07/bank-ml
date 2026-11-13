# Bank Marketing - Tiền xử lý dữ liệu

## 1. Giới thiệu

Dự án thực hiện quy trình tiền xử lý cho bộ dữ liệu **Bank Marketing**. Mục tiêu là chuẩn bị dữ liệu cho các mô hình Machine Learning dự đoán khách hàng có đăng ký tiền gửi kỳ hạn hay không.

Biến mục tiêu:

- `y = 0`: khách hàng không đăng ký.
- `y = 1`: khách hàng đăng ký.

Notebook chính: [pre_processing.ipynb](./pre_processing.ipynb)

## 2. Nội dung thực hiện

Notebook và script gồm các bước:

1. Tải dữ liệu từ `bank-full.csv`, kiểm tra kiểu dữ liệu và xóa trùng lặp (`drop_duplicates()`).
2. Ánh xạ tĩnh (mapping) các biến nhị phân (`default`, `housing`, `loan`, `y`), thứ tự (`education`) và thời gian (`month`).
3. Tách dữ liệu `train_test_split(test_size=0.2, random_state=42, stratify=y)` sớm nhất để ngăn ngừa rò rỉ dữ liệu (Data Leakage).
4. Phân tích khám phá (EDA) và trực quan hóa tỷ lệ missing/unknown và boxplot ngoại lệ chỉ trên tập Train.
5. Xây dựng `ColumnTransformer` tích hợp `SimpleImputer`, `IQRCapper` (custom transformer), `RobustScaler`, và `OneHotEncoder`: fit trên tập Train, sau đó transform cho cả Train và Test.
6. Trực quan hóa phân phối, số chiều sau mã hóa và ma trận tương quan chỉ dựa trên tập Train.
7. Cân bằng lớp bằng `SMOTE` chỉ áp dụng trên tập Train đã xử lý (`X_train_processed`, `y_train`). Giữ nguyên phân phối thực tế của tập Test.
8. Xuất dữ liệu đã xử lý sạch ra các file CSV riêng biệt (`train_processed.csv`, `train_resampled.csv`, `test_processed.csv`).

## 3. Cấu trúc thư mục

```text
ml_assign/
├── bank-full.csv
├── pre_processing.ipynb
├── pre_processing.py
├── train_processed.csv
├── train_resampled.csv
├── test_processed.csv
├── outputs/
│   └── plots/
└── README.md
```

`train_processed.csv`, `train_resampled.csv`, và `test_processed.csv` được tạo sau khi chạy pipeline thành công.

## 4. Yêu cầu môi trường

- Python 3.10 trở lên
- Jupyter Notebook hoặc JupyterLab
- Các thư viện:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn

## 5. Cài đặt thư viện

Mở PowerShell tại thư mục dự án và chạy:

```powershell
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Nếu dự án sử dụng virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

## 6. Hướng dẫn chạy

### Chạy bằng script Python

Mở PowerShell tại thư mục dự án và chạy:

```powershell
python pre_processing.py
```

Script thực hiện cùng pipeline với notebook, hiển thị các bảng/biểu đồ, áp dụng SMOTE trên tập train và tạo `cleaned_bank_data.csv`. Các biểu đồ được lưu tự động trong `outputs/plots/`.

### Chạy bằng Jupyter Notebook

Từ thư mục `ml_assign`, chạy:

```powershell
jupyter notebook
```

Sau đó:

1. Mở file `pre_processing.ipynb`.
2. Chọn đúng Python kernel đã cài các thư viện.
3. Chạy lần lượt các cell từ trên xuống dưới bằng `Run All`.
4. Kiểm tra các bảng và biểu đồ kết quả.

### Chạy bằng JupyterLab

```powershell
jupyter lab
```

Mở `pre_processing.ipynb` và chạy toàn bộ notebook.

## 7. Xử lý mất cân bằng lớp và Data Leakage

Dữ liệu ban đầu có tỷ lệ khoảng:

- `no`: 88%.
- `yes`: 12%.

Notebook thực hiện tách train/test trước:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

SMOTE chỉ được áp dụng trên tập train:

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

Không áp dụng SMOTE lên `X_test` hoặc `y_test`. Tập test phải giữ phân phối thực tế để đánh giá mô hình khách quan. Khi huấn luyện mô hình, nên ưu tiên F1-Score, Precision, Recall và PR-AUC thay vì chỉ sử dụng Accuracy.

## 8. Kết quả đầu ra

Sau khi chạy thành công, pipeline tạo các file dữ liệu sạch sẵn sàng cho bước huấn luyện:

- `train_processed.csv`: Tập train (36,168 dòng x 35 cột) đã qua pipeline tiền xử lý (giữ nguyên phân phối mất cân bằng).
- `train_resampled.csv`: Tập train (63,874 dòng x 35 cột) đã được cân bằng lớp 50:50 bằng SMOTE.
- `test_processed.csv`: Tập test (9,043 dòng x 35 cột) được biến đổi bằng các tham số học từ tập train, giữ nguyên nhãn gốc để đánh giá khách quan.

Các file dữ liệu đã:
- Xử lý giá trị thiếu bằng median/mode học từ train.
- Capping các giá trị ngoại lệ theo ngưỡng IQR học từ train.
- Mã hóa One-Hot các biến danh mục và ánh xạ các biến thứ tự/nhị phân.
- Chuẩn hóa các biến số bằng `RobustScaler` học từ train.

Nếu bất kỳ file CSV nào đang mở trong Excel hoặc chương trình khác, hãy đóng file trước khi chạy để tránh lỗi `PermissionError`.

Các biểu đồ được xuất dưới dạng PNG với tên theo thứ tự pipeline:

```text
outputs/plots/
├── 01_missing_unknown_rate.png
├── 02_outliers_before_capping.png
├── 03_outliers_after_capping.png
├── 04_features_before_after_encoding.png
├── 05_distributions_before_after_scaling.png
├── 06_correlation_heatmap.png
├── 07_target_distribution_before_smote.png
└── 08_train_distribution_before_after_smote.png
```
