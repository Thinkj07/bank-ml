# Bank Marketing - Bài tập lớn Học máy (CO3117)

Dự đoán khách hàng có đăng ký tiền gửi kỳ hạn hay không (biến mục tiêu `y`, 1 là có). Dữ liệu: [Bank Marketing, UCI](https://archive.ics.uci.edu/dataset/222/bank), file `bank-full.csv` (45.211 dòng). Chỉ khoảng 11,7% khách đồng ý, nên bài toán mất cân bằng lớp. Cột `duration` bị loại vì chỉ biết sau cuộc gọi.

## Thành viên

| Họ tên | MSSV | Phần phụ trách |
|---|---|---|
| Nguyễn Đình Khang | 2052519 | Phần D: Gradient Boosting, feature selection, MLP, so sánh mô hình |
| Nguyễn Tuấn Huy | 2311223 | Phần B: Logistic Regression, SVM |
| [điền họ tên] | [điền MSSV] | Phần A: dữ liệu và tiền xử lý |
| [điền họ tên] | [điền MSSV] | Phần C: Random Forest |

## Kết quả chính

Đánh giá trên tập test (9.043 dòng, 1.058 khách đồng ý). Ngưỡng quyết định được chọn trên train, test chỉ dùng để báo cáo.

| Mô hình | F1 | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0,390 | 0,366 | 0,756 | 0,356 | 0,432 |
| MLP | 0,428 | 0,405 | 0,768 | 0,427 | 0,429 |
| Random Forest | 0,482 | 0,453 | 0,801 | 0,431 | 0,547 |
| Gradient Boosting | 0,501 | 0,472 | 0,808 | 0,461 | 0,549 |
| Gradient Boosting, 15 đặc trưng | 0,498 | 0,469 | 0,804 | 0,468 | 0,532 |

Bảng đầy đủ nằm ở `reports/results/comparison.csv`, bảng gọi top k phần trăm khách ở `reports/results/topk.csv`, mọi lần chạy ở `reports/results/results.csv`.

## Cấu trúc thư mục

```text
configs/config.yaml     tham số dùng chung (seed, tỷ lệ chia, drop_duration)
data/raw/               dữ liệu gốc
data/processed/         train.csv và test.csv đã chia sẵn, dùng chung cho cả nhóm
src/                    code dùng lại: preprocessing, data, evaluate, tuning, plots, models
src/models/             logreg.py, rf.py, gb.py, mlp.py (mỗi file có build và search_space)
scripts/                make_dataset, train_rf, train_gb, train_mlp, feature_selection,
                        compare_models, smote_experiment, bootstrap_ci
notebooks/              02_model_logreg.ipynb (Logistic Regression)
models/                 mô hình đã huấn luyện (tạo khi chạy script, không commit)
reports/                figures, results, report, slides
docs/                   kế hoạch và ghi chú nhóm
legacy/                 bản tiền xử lý cũ, giữ để tham khảo
```

## Cài đặt

Cần Python 3.12 trở lên (đã chạy với 3.13). Số liệu trong báo cáo được tạo với scikit-learn 1.9.1; các phiên bản khác có thể cho kết quả hơi khác, nên `requirements.txt` ghim đúng phiên bản.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows PowerShell, kích hoạt bằng `.venv\Scripts\Activate.ps1`.

## Dữ liệu train/test dùng chung

`data/processed/train.csv` và `data/processed/test.csv` đã nằm sẵn trong repo, cả nhóm dùng đúng hai file này. Không tự chia lại dữ liệu.

```python
from src.data import load_train_test

X_train, X_test, y_train, y_test = load_train_test()
```

Cách tạo hai file:

1. Đọc dữ liệu gốc, bỏ dòng trùng, mã hóa biến phân loại.
2. Chia train/test 80/20 với `stratify=y` và `random_state=42`.
3. Fit việc cắt ngoại lệ IQR và `RobustScaler` chỉ trên train, sau đó áp dụng sang test.

Nếu muốn tạo lại, chạy `python -m scripts.make_dataset`. Việc này làm thay đổi dữ liệu nên mọi mô hình phải huấn luyện lại.

SMOTE không nằm trong file. Khi dùng, SMOTE chỉ áp dụng trên train, và trong cross-validation phải đặt SMOTE trong `imblearn.pipeline.Pipeline` để nó chỉ chạy trên fold train.

## Quy trình đánh giá

Mọi mô hình dùng chung một quy trình để so sánh công bằng:

- Tuning bằng `RandomizedSearchCV`, cross-validation 5 fold có stratify, chỉ trên train, chấm bằng PR-AUC (Random Forest do một thành viên khác làm nên chọn tham số bằng F1).
- Ngưỡng quyết định chọn để F1 lớn nhất trên dự đoán out-of-fold của train, rồi áp sang test.
- Test chỉ dùng để báo cáo, không dùng để chọn tham số hay ngưỡng.
- Thước đo chính: F1 và PR-AUC. Accuracy không dùng làm tiêu chí vì đoán toàn lớp 0 đã đạt khoảng 88%.
- Bảng phụ cho góc nhìn kinh doanh: gọi top 5, 10, 20, 30% khách có điểm cao nhất, xem precision, recall và lift so với gọi ngẫu nhiên.

```python
from src.tuning import run_experiment

model, metrics = run_experiment("ten_mo_hinh", estimator, X_train, y_train, X_test, y_test)
```

Hàm này huấn luyện, chọn ngưỡng, đánh giá trên test và ghi một dòng vào `reports/results/results.csv` (dòng cùng tên sẽ bị ghi đè).

## Chạy lại thí nghiệm

Chạy từ thư mục gốc của repo, theo thứ tự dưới đây (các bước sau cần file `*_best_params.json` do bước trước tạo ra).

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_logreg.ipynb
python -m scripts.train_rf
python -m scripts.train_gb
python -m scripts.feature_selection
python -m scripts.train_mlp
python -m scripts.compare_models
```

| Bước | Việc làm | Đầu ra chính |
|---|---|---|
| `02_model_logreg.ipynb` | Logistic Regression: baseline, đổi ngưỡng, tuning, SMOTE | `logreg_best_params.json`, `logreg_*.png` |
| `train_rf` | Random Forest: baseline, tuning, đổi ngưỡng, độ quan trọng đặc trưng | `rf_best_params.json`, `rf_*.csv`, `rf_*.png`, `models/rf_tuned.joblib` |
| `train_gb` | Gradient Boosting: baseline, tuning, đổi ngưỡng | `gb_best_params.json`, `gb_*.png` |
| `feature_selection` | Xếp hạng đặc trưng bằng permutation importance trên train, chọn số đặc trưng bằng cross-validation, huấn luyện lại GB và RF | `fs_*.csv`, `fs_selected.json`, `fs_*.png` |
| `train_mlp` | MLP baseline | `mlp_best_params.json`, `mlp_*.png` |
| `compare_models` | Huấn luyện lại các mô hình từ file tham số, bảng so sánh, top k, biểu đồ chung | `comparison.csv`, `topk.csv`, `compare_*.png` |

Hai thí nghiệm bổ sung (chạy sau khi đã có `rf_best_params.json` và `gb_best_params.json`):

```bash
python -m scripts.smote_experiment
python -m scripts.bootstrap_ci
```

- `smote_experiment`: so sánh Random Forest có `class_weight`, không `class_weight` và có SMOTE (kết quả ở `smote_rf.csv`).
- `bootstrap_ci`: bootstrap 1000 lần trên tập test để tính khoảng tin cậy 95% của F1, PR-AUC, ROC-AUC của Random Forest và Gradient Boosting, và của hiệu hai mô hình (kết quả ở `bootstrap_ci.csv`).

Random Forest tuning là bước chạy lâu nhất. Seed cố định là 42 trong `configs/config.yaml`.

## Thêm mô hình mới

Mô hình mới đặt trong `src/models/<ten>.py` với hàm `build(...)` trả về mô hình chưa huấn luyện (tham số tường minh) và hàm `search_space()` cho tuning, rồi viết script tương tự `scripts/train_gb.py`.

## Ghi chú về `duration`

`configs/config.yaml` có `drop_duration: true`. Cột `duration` (thời lượng cuộc gọi) chỉ biết sau khi cuộc gọi kết thúc nên được loại bỏ, nếu giữ lại sẽ là rò rỉ dữ liệu. Đổi giá trị này phải được cả nhóm thống nhất rồi chạy lại `make_dataset`. Điểm số thấp hơn các kết quả công bố có dùng `duration` là điều bình thường.
