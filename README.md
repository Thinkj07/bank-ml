# Bank Marketing - Tiền xử lý dữ liệu

## 1. Giới thiệu

Dự án thực hiện quy trình tiền xử lý cho bộ dữ liệu **Bank Marketing**. Mục tiêu là chuẩn bị dữ liệu cho các mô hình Machine Learning dự đoán khách hàng có đăng ký tiền gửi kỳ hạn hay không.

Biến mục tiêu:

- `y = 0`: khách hàng không đăng ký.
- `y = 1`: khách hàng đăng ký.

Notebook chính: [pre_processing.ipynb](./pre_processing.ipynb)

## 2. Nội dung thực hiện

Notebook gồm các bước:

1. Tải dữ liệu từ `bank-full.csv`
2. Kiểm tra cấu trúc dữ liệu, kích thước và kiểu dữ liệu.
3. Kiểm tra giá trị thiếu `NaN/Null` và giá trị ẩn `unknown`.
4. Xử lý dữ liệu trùng lặp.
5. Phát hiện và capping ngoại lệ bằng phương pháp IQR.
6. Mã hóa biến phân loại
7. Chuẩn hóa các biến số bằng `RobustScaler`.
8. Phân tích tương quan và đa cộng tuyến.
9. Tách dữ liệu train/test với `stratify=y`.
10. Cân bằng lớp bằng SMOTE chỉ trên tập train để tránh Data Leakage.
11. Xuất dữ liệu đã xử lý ra file CSV.

## 3. Cấu trúc thư mục

```text
ml_assign/
├── bank-full.csv
├── pre_processing.ipynb
├── cleaned_bank_data.csv
└── README.md
```

`cleaned_bank_data.csv` được tạo sau khi chạy notebook thành công.

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

### Cách 1: Chạy bằng Jupyter Notebook

Từ thư mục `ml_assign`, chạy:

```powershell
jupyter notebook
```

Sau đó:

1. Mở file `pre_processing.ipynb`.
2. Chọn đúng Python kernel đã cài các thư viện.
3. Chạy lần lượt các cell từ trên xuống dưới bằng `Run All`.
4. Kiểm tra các bảng và biểu đồ kết quả.

### Cách 2: Chạy bằng JupyterLab

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

Sau khi chạy thành công, notebook tạo:

```text
cleaned_bank_data.csv
```

File này chứa dữ liệu đã:

- Xử lý giá trị thiếu.
- Giữ lại các nhóm `unknown` có ý nghĩa.
- Xử lý dòng trùng lặp.
- Capping một số ngoại lệ theo IQR.
- Mã hóa biến phân loại.
- Chuẩn hóa biến số.

Nếu `cleaned_bank_data.csv` đang mở trong Excel hoặc chương trình khác, hãy đóng file trước khi chạy cell xuất dữ liệu để tránh lỗi `PermissionError`.
