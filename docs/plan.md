# Kế hoạch Bài tập lớn Học máy (CO3117)

Nhóm 4 người. Tài liệu này tóm tắt yêu cầu đề bài, cách chia việc và mốc thời gian để cả nhóm thống nhất.

## 1. Mục tiêu

Chạy trọn một quy trình học máy trên một bài toán thực tế: tiền xử lý dữ liệu, chọn đặc trưng, chọn mô hình, tối ưu tham số, đánh giá bằng nhiều metric. Đề nhấn mạnh việc áp dụng đúng và hiểu rõ kỹ thuật, không yêu cầu mô hình phức tạp hay kết quả vượt SOTA. Mỗi lựa chọn (dataset, metric, thuật toán) đều phải giải thích được lý do.

## 2. Yêu cầu bắt buộc

1. Chọn dataset (UCI, Kaggle, danh sách trên Wikipedia). Nếu dùng nguồn khác mà chưa chắc thì phải hỏi giảng viên duyệt. Nêu rõ vì sao phù hợp với ML và thách thức chính của dữ liệu.
2. Phân tích bài toán và dữ liệu: dạng dữ liệu, số mẫu, phân bố, các khó khăn có thể gặp.
3. Áp dụng ít nhất 2 thuật toán khác nhau.
4. Thực nghiệm, so sánh, phân tích kết quả. Giải thích vì sao chọn metric đó.
5. Nên thử thêm: feature selection, hyperparameter tuning, ensemble, baseline Deep Learning.
6. Viết báo cáo. Outline gợi ý: Giới thiệu bài toán, Dữ liệu và tiền xử lý, Mô hình và phương pháp, Thực nghiệm và kết quả, Thảo luận, Kết luận và hướng phát triển.
7. Chuẩn bị slide thuyết trình.

## 3. Yêu cầu nộp bài

- Một file zip tên dạng MSSV1-MSSV2-MSSV3-MSSV4.zip gồm source code, data, model, báo cáo, slide, Dockerfile nếu có.
- Bắt buộc có README.md mô tả cách chạy code, môi trường, ghi chú cần thiết.
- Nên có requirements.txt, environment.yml hoặc Dockerfile. Giảng viên có thể chạy lại code để kiểm tra tính tái hiện.
- File lớn (data, model) upload lên Google Drive hoặc GitHub, đưa link public vào README.

## 4. Lưu ý về gian lận

- Mọi thành viên phải tự tham gia thực hiện.
- Được dùng thư viện open-source, không được dùng API thương mại thay thế toàn bộ pipeline.
- Không chia sẻ hoặc sao chép code giữa các nhóm.
- Được dùng AI để tham khảo, giải thích, debug. Không được copy nguyên báo cáo hoặc code mà không hiểu. Mỗi người chịu trách nhiệm về nội dung nộp, nên ai cũng cần hiểu toàn bộ pipeline.

## 5. Phân công

Chia theo module để làm song song, nhưng mỗi người đều viết code và báo cáo cho phần của mình.

| Thành viên | Phụ trách | Sản phẩm |
|---|---|---|
| A (Dataset + Preproc) | Mô tả dataset, EDA, tiền xử lý (missing, encoding, scaling, mất cân bằng), pipeline dữ liệu dùng chung | Notebook EDA, module preprocessing, mục 1 và 2 của báo cáo |
| B (mô hình nhóm 1) | Thuật toán thứ nhất (ví dụ Logistic Regression hoặc SVM) làm baseline, kèm tuning | Code train/eval, bảng kết quả, mô tả mô hình |
| C (mô hình nhóm 2) | Thuật toán thứ hai (ví dụ Random Forest hoặc Gradient Boosting), kèm tuning | Code train/eval, bảng kết quả, mô tả mô hình |
| D (nâng cao và tổng hợp) | Feature selection, ensemble hoặc baseline DL; script so sánh mô hình; biểu đồ (confusion matrix, ROC, learning curve); phân tích lỗi | Mục 4 và 5 của báo cáo, hình minh họa |

Việc chung của cả nhóm:

- Giai đoạn 0 (cả nhóm): chốt dataset, bài toán, metric chính, cách chia train/val/test, random seed cố định, cấu trúc repo.
- Báo cáo: mỗi người viết phần mô hình và kết quả của mình, A rà soát văn phong và định dạng.
- Slide: mỗi người làm phần của mình, cả nhóm thống nhất lại.
- Đóng gói: D viết README.md và requirements.txt, A chạy lại từ đầu trên môi trường sạch để kiểm tra.
- Review chéo code: B xem code của C và ngược lại, A xem code của D và ngược lại.

## 6. Mốc thời gian (điều chỉnh theo deadline thực tế)

| Mốc | Công việc |
|---|---|
| Tuần 1 | Chốt dataset, hoàn thành EDA và tiền xử lý |
| Tuần 2 | Chạy xong hai mô hình chính, có bảng so sánh đầu tiên |
| Tuần 3 | Phần nâng cao, viết báo cáo, chuẩn bị slide |
| Vài ngày cuối | Ghép báo cáo, chạy lại toàn bộ trên máy sạch, đóng gói zip |

## 7. Checklist trước khi nộp

- [ ] Dataset đã được giảng viên duyệt (nếu ngoài danh sách gợi ý)
- [ ] Ít nhất 2 thuật toán, có so sánh và giải thích metric
- [ ] Báo cáo đủ các mục trong outline
- [ ] Slide hoàn chỉnh
- [ ] README.md và file môi trường (requirements.txt / environment.yml / Dockerfile)
- [ ] File lớn đã upload và có link public trong README
- [ ] Code chạy lại được từ đầu trên môi trường sạch
- [ ] Tên file zip đúng dạng MSSV1-MSSV2-MSSV3-MSSV4.zip
