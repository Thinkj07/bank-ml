# Random Forest (thành viên C)

Bản nháp cho mục "Mô hình và phương pháp" và "Thực nghiệm và kết quả" của báo cáo.

## 1. Vì sao chọn Random Forest

- Dữ liệu dạng bảng, trộn biến số và biến phân loại đã mã hóa (33 cột sau one-hot). Cây quyết định không cần giả định tuyến tính, tự bắt được tương tác giữa các biến (ví dụ `month` × `poutcome`), và không nhạy với việc scale.
- Random Forest gộp nhiều cây huấn luyện trên mẫu bootstrap và tập con đặc trưng ngẫu nhiên, nên giảm phương sai so với một cây đơn, ít overfit hơn.
- Là mô hình ensemble (bagging), bổ sung tốt cho baseline tuyến tính (Logistic Regression của B) để so sánh.
- Có sẵn độ quan trọng đặc trưng để phân tích.

## 2. Thiết lập thực nghiệm

- Dữ liệu: `data/processed/train.csv` (36 168 dòng) và `test.csv` (9 043 dòng) dùng chung cả nhóm, tỷ lệ lớp "yes" khoảng 11,7%. Đã bỏ `duration`.
- Mất cân bằng lớp: xử lý bằng `class_weight` (`balanced` / `balanced_subsample`) thay vì SMOTE, để không tạo mẫu tổng hợp và không phải đưa SMOTE vào pipeline CV. `class_weight` là một tham số được tuning, nên CV tự chọn có dùng hay không.
- Metric chọn mô hình: F1 của lớp dương (metric chính của nhóm). Báo thêm precision, recall, PR-AUC, ROC-AUC. Accuracy không phù hợp làm metric chính vì đoán toàn "no" đã được khoảng 88%.
- Ba cấu hình:
  1. `rf_baseline`: 100 cây, tham số mặc định, không xử lý mất cân bằng.
  2. `rf_tuned`: `RandomizedSearchCV` với `StratifiedKFold` (shuffle, seed 42), chấm điểm bằng F1. Không gian tìm kiếm: `n_estimators` 100–500, `max_depth` {None, 8, 12, 16, 20, 25, 30}, `min_samples_split` 2–20, `min_samples_leaf` 1–10, `max_features` {sqrt, log2, 0.5}, `class_weight` {None, balanced, balanced_subsample}.
  3. `rf_tuned_threshold`: mô hình tuned, đổi ngưỡng quyết định từ 0,5 sang ngưỡng cho F1 cao nhất trên xác suất out-of-fold của tập train (`cross_val_predict`). Ngưỡng chọn hoàn toàn trên train, tập test chỉ dùng để báo cáo.

## 3. Kết quả trên tập test

Kết quả dưới đây từ lần chạy kiểm tra rút gọn (`--n-iter 8 --cv 3`). Chạy bản đầy đủ (`python -m scripts.train_rf`, 30 tổ hợp × 5 fold) rồi cập nhật lại bảng từ `reports/results/rf_comparison.csv`.

| Mô hình | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| rf_baseline | 0.8915 | 0.6097 | 0.2023 | 0.3038 | 0.4133 | 0.7810 |
| rf_tuned | 0.8686 | 0.4466 | 0.5142 | 0.4780 | 0.4533 | 0.8013 |
| rf_tuned_threshold | 0.8658 | 0.4390 | 0.5302 | 0.4803 | 0.4533 | 0.8013 |

Tham số tốt nhất (lần chạy rút gọn): `class_weight=balanced_subsample`, `max_depth=30`, `max_features=0.5`, `min_samples_leaf=8`, `min_samples_split=5`, `n_estimators=459`. F1 CV 0,4677; ngưỡng 0,48.

Nhận xét:

- Baseline có accuracy cao nhất nhưng chỉ bắt được khoảng 20% khách "yes" (214/1058): mô hình thiên về lớp đa số.
- Tuning nâng F1 từ 0,30 lên 0,48 và recall từ 0,20 lên 0,51, đổi lại precision giảm (0,61 xuống 0,45) và accuracy giảm nhẹ. Phần lớn cải thiện đến từ `class_weight` và `min_samples_leaf` lớn (lá to hơn, xác suất mượt hơn).
- PR-AUC và ROC-AUC cũng tăng, nên mô hình tuned xếp hạng khách hàng tốt hơn, không chỉ dịch ngưỡng.
- Chỉnh ngưỡng chỉ thêm rất ít (F1 0,478 lên 0,480), vì `class_weight` đã đẩy xác suất về phía lớp dương; ngưỡng tối ưu gần 0,5.
- Trong bối cảnh telemarketing, bỏ sót khách sẽ đăng ký (FN) thường tốn hơn gọi nhầm (FP), nên đánh đổi precision lấy recall là hợp lý. Nếu ngân hàng muốn ít cuộc gọi hơn, có thể nâng ngưỡng.

## 4. Đặc trưng quan trọng

Hình `reports/figures/rf_feature_importance.png` so sánh hai cách đo:

- MDI (impurity) xếp `balance`, `age`, `day` lên đầu. MDI thiên vị các biến liên tục có nhiều điểm chia, nên không đáng tin hoàn toàn.
- Permutation importance trên test (mức giảm PR-AUC khi xáo trộn một cột) xếp `poutcome_success`, `month`, `day`, `housing`, `contact_unknown`, `pdays` lên đầu. `balance` và `age` giảm mạnh thứ hạng.
- Kết luận: kết quả chiến dịch trước (`poutcome_success`, `pdays`) và thời điểm liên hệ (`month`, `day`) là tín hiệu mạnh nhất; nhóm nghề nghiệp và học vấn đóng góp ít.

## 5. Hạn chế và hướng tiếp theo

- Random search chỉ thử 30 tổ hợp; có thể thử thêm Gradient Boosting (LightGBM/XGBoost) để so sánh trong phần ensemble của D.
- F1 khoảng 0,48 phản ánh độ khó của bài toán khi đã bỏ `duration`.
- Có thể dùng permutation importance để loại các đặc trưng gần 0 rồi huấn luyện lại (feature selection).

## 6. Cách chạy lại

```bash
python -m scripts.train_rf               # đầy đủ: 30 tổ hợp x 5 fold
python -m scripts.train_rf --n-iter 10   # nhanh hơn
```

Đầu ra: `models/rf_tuned.joblib`, `reports/results/rf_*.csv|json`, `reports/figures/rf_*.png`, và ba dòng `rf_*` trong `reports/results/results.csv` (chạy lại sẽ thay các dòng cũ, không bị trùng).
