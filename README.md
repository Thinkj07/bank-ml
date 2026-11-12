# Bank Marketing - Bài tập lớn Học máy (CO3117)

Dự đoán khách hàng có đăng ký tiền gửi kỳ hạn hay không (biến mục tiêu `y`, 1 là có). Dữ liệu: [Bank Marketing, UCI](https://archive.ics.uci.edu/dataset/222/bank), file `bank-full.csv`.

## Cấu trúc thư mục

```text
configs/config.yaml     tham số dùng chung (seed, tỷ lệ chia, drop_duration)
data/raw/               dữ liệu gốc
data/processed/         train.csv và test.csv đã chia sẵn, dùng chung cho cả nhóm
src/                    code dùng lại: preprocessing, data, evaluate, models
scripts/                make_dataset.py tạo train/test
notebooks/              mỗi người một notebook riêng
models/                 model đã huấn luyện (không commit)
reports/                figures, results, report, slides
docs/                   kế hoạch và tài liệu nhóm
legacy/                 bản tiền xử lý cũ, giữ để tham khảo
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows PowerShell, kích hoạt bằng `.venv\Scripts\Activate.ps1`.

## Dữ liệu train/test dùng chung

`data/processed/train.csv` và `data/processed/test.csv` đã được commit, cả nhóm dùng đúng hai file này. Không tự chia lại dữ liệu.

```python
from src.data import load_train_test

X_train, X_test, y_train, y_test = load_train_test()
```

Quy trình tạo hai file:

1. Đọc dữ liệu gốc, bỏ dòng trùng, mã hóa biến phân loại.
2. Chia train/test 80/20 với `stratify=y` và `random_state=42`.
3. Fit việc cắt ngoại lệ IQR và `RobustScaler` chỉ trên train, sau đó áp dụng sang test.

SMOTE không nằm trong file. Chỉ áp dụng SMOTE trên train, và nếu dùng cross-validation thì phải đặt SMOTE trong `imblearn.pipeline.Pipeline` để nó chỉ chạy trên fold train:

```python
from src.data import smote_resample

X_res, y_res = smote_resample(X_train, y_train)
```

Chỉ A được tạo lại hai file này bằng `python -m scripts.make_dataset`, và phải báo cả nhóm để mọi người huấn luyện lại.

## Đánh giá mô hình

```python
from src.evaluate import evaluate, log_result

metrics = evaluate(model, X_test, y_test)
log_result("logreg_baseline", metrics)
```

Kết quả được ghi vào `reports/results/results.csv`, dùng chung để so sánh mô hình.

## Quy ước làm việc

- Không commit trực tiếp lên `main`. Mỗi người làm trên nhánh riêng (`feat/a-preprocessing`, `feat/b-logreg`, `feat/c-rf`, `feat/d-advanced`) và mở Pull Request cho người khác review.
- Cài `nbstripout` một lần để notebook không lưu output: `nbstripout --install`.
- Mô hình mới thêm vào `src/models/<ten>.py` với hàm `build(**params)`.
- Metric chính: F1 của lớp dương, báo thêm precision, recall, PR-AUC, ROC-AUC.

## Ghi chú về `duration`

`configs/config.yaml` có `drop_duration: true`. Cột `duration` (thời lượng cuộc gọi) chỉ biết sau khi cuộc gọi kết thúc nên mặc định được loại bỏ. Đổi giá trị này phải được cả nhóm thống nhất rồi chạy lại `make_dataset`.
