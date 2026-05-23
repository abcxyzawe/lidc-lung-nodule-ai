# Mục Tiêu và Phạm Vi Nghiên Cứu

**Luận văn**: Hệ thống phát hiện và phân loại nốt phổi tự động hỗ trợ đọc ảnh CT lồng ngực trên tập dữ liệu LIDC-IDRI  
**Phiên bản**: 2026-05-23

---

## 1. Tính Cấp Thiết của Đề Tài

Ung thư phổi là một trong những nguyên nhân gây tử vong hàng đầu do ung thư tại Việt Nam và trên toàn thế giới. Theo ghi nhận của GLOBOCAN, ung thư phổi thuộc nhóm hai loại ung thư có tỷ lệ tử vong cao nhất, đồng thời là loại ung thư có tiên lượng rất xấu khi chẩn đoán ở giai đoạn muộn. Nghiên cứu thử nghiệm sàng lọc phổi quốc gia Hoa Kỳ (National Lung Screening Trial — NLST, 2011) đã chứng minh rằng sàng lọc định kỳ bằng chụp cắt lớp vi tính liều thấp (low-dose CT) ở nhóm nguy cơ cao giúp giảm 20% tỷ lệ tử vong do ung thư phổi so với chụp X-quang ngực thông thường [NLST Research Team, 2011]. Kết quả này xác lập vai trò trung tâm của CT ngực trong tầm soát ung thư phổi sớm.

Tuy nhiên, việc đọc và phân tích ảnh CT ngực trong tầm soát đại trà đặt ra thách thức lớn đối với hệ thống y tế. Một ca CT ngực tiêu chuẩn có thể bao gồm hàng trăm lát cắt, đòi hỏi bác sĩ X-quang phải duy trì sự tập trung liên tục trong suốt quá trình đọc. Khi khối lượng công việc tăng cao, sự mệt mỏi (reader fatigue) và tính biến động giữa các bác sĩ (inter-reader variability) trở thành yếu tố ảnh hưởng đến độ nhạy phát hiện nốt. Tình trạng thiếu bác sĩ X-quang có chuyên môn, đặc biệt tại các cơ sở y tế tuyến tỉnh và khu vực ngoại thành ở Việt Nam, càng làm trầm trọng thêm khoảng cách giữa nhu cầu tầm soát và năng lực đọc kết quả. Trong bối cảnh đó, trí tuệ nhân tạo (AI) có tiềm năng đóng vai trò công cụ hỗ trợ đọc thứ hai (AI second reader) — không thay thế bác sĩ, mà nâng cao tính nhất quán và hỗ trợ phát hiện các trường hợp dễ bị bỏ sót, từ đó giúp bác sĩ X-quang tập trung quyết định lâm sàng vào những ca khó.

---

## 2. Mục Tiêu Nghiên Cứu

### 2.1 Mục Tiêu Nghiên Cứu

Đồ án hướng tới việc nghiên cứu, thiết kế và xây dựng hoàn chỉnh một hệ thống phần mềm hỗ trợ phát hiện và phân loại nốt phổi thông minh, ứng dụng sâu rộng các kỹ thuật Trí tuệ nhân tạo (AI) và học sâu y khoa để tự động hóa quy trình tầm soát ung thư phổi trên ảnh chụp cắt lớp vi tính lồng ngực. Để hiện thực hóa tầm nhìn này, đề tài phân rã thành các mục tiêu cốt lõi và cụ thể như sau:

**Nghiên cứu cơ sở lý thuyết và kiến trúc lõi của thị giác máy tính trong ảnh y khoa**: Tổng hợp, hệ thống hóa và đi sâu phân tích nền tảng của bài toán phân đoạn ảnh y khoa ba chiều. Trọng tâm nghiên cứu được đặt vào việc giải phẫu và làm chủ kiến trúc mạng nơ-ron tích chập (CNN) đa tầng, đặc biệt là họ mô hình U-Net cải tiến (UNet++) kết hợp encoder EfficientNet-B5 cùng cơ chế chú ý không gian-kênh đồng thời (SCSE — Spatial and Channel Squeeze-Excitation). Đánh giá ưu/nhược điểm của kiến trúc 2.5D segmentation đề xuất so với kiến trúc 3D detection tiêu chuẩn (MONAI RetinaNet 3D) nhằm chứng minh sự phù hợp của nó với đặc thù dữ liệu CT lồng ngực thưa-dày bất đồng nhất.

**Xây dựng, huấn luyện và tối ưu hóa pipeline phát hiện nốt phổi đa tầng (4-Stage Detection Pipeline)**: Tổ chức xử lý, chuẩn hóa và xây dựng nhãn đồng thuận trên tập dữ liệu LIDC-IDRI quy mô lớn (1.010 bệnh nhân, hơn 200.000 lát cắt DICOM, với annotation từ bốn bác sĩ X-quang đọc độc lập). Tiến hành huấn luyện tuần tự bốn mô hình: phân đoạn khởi tạo Stage 1, phân đoạn tinh chỉnh Stage 2 với hàm mất mát Tversky bất đối xứng kết hợp Stochastic Weight Averaging (SWA), bộ lọc dương tính giả DenseNet121-3D, và bộ phân loại mức độ ác tính 5 lớp. Tinh chỉnh siêu tham số và đánh giá mô hình bằng các thang đo chuẩn xác như Dice, F1, FROC và CPM (Competition Performance Metric), từ đó thiết lập một đường ống phát hiện đạt độ tin cậy cực cao trên tập kiểm thử bị khóa (val_dice = 0.8676, F1 = 0.618 trên 99 bệnh nhân kiểm thử).

**Phát triển các thuật toán Tiền xử lý và Hậu xử lý không gian nâng cao**: Nhận thức được rào cản của môi trường ảnh y khoa thực tế, đề tài đặt mục tiêu xây dựng các module xử lý ảnh truyền thống để "bọc lót" cho AI. Về tiền xử lý, tập trung vào kỹ thuật chuẩn hóa cửa sổ HU (Hounsfield Unit) trong dải [-1350, 150] và xây dựng đầu vào 2.5D ba kênh từ các lát cắt lân cận để bù đắp thông tin chiều sâu cho mô hình 2D. Về hậu xử lý, phát triển thuật toán lọc hình thái học theo tỷ lệ kéo dài (elongation ratio) nhằm loại bỏ các cấu trúc mạch máu cắt ngang, kết hợp thuật toán Non-Maximum Suppression (NMS) ba chiều để gộp các nốt chồng lấn, và ngưỡng FPR học từ dữ liệu (best_thr = 0.85, AUC = 0.9364) để loại bỏ dương tính giả trước khi trả về cho bác sĩ.

**Phát triển kiến trúc phần mềm và hệ webapp đa chế độ toàn trình (End-to-End)**: Thiết kế giao diện người dùng trực quan, thân thiện bằng nền tảng ứng dụng Web (Next.js + FastAPI), cho phép bác sĩ X-quang thao tác dễ dàng — tải lên ca bệnh DICOM, theo dõi tiến trình phân tích thời gian thực qua Server-Sent Events, và đối chiếu kết quả ba nguồn (mô hình đề xuất / MONAI RetinaNet 3D / ground truth từ chuyên gia LIDC) trên cùng một trang. Xây dựng cấu trúc dữ liệu chuẩn hóa để lưu trữ và truy xuất phán quyết lâm sàng (chấp nhận / từ chối / yêu cầu xem lại) đồng thời theo từng nốt và từng chế độ AI, hỗ trợ truy vết audit và phân tích hậu kỳ.

**Tự động hóa luồng nghiệp vụ hỗ trợ đọc lâm sàng**: Mục tiêu cuối cùng là đóng gói các công nghệ trên thành một quy trình nghiệp vụ khép kín mô phỏng "AI second reader". Bao gồm: tiếp nhận ca bệnh DICOM, tự động phát hiện và đo kích thước nốt phổi, phân loại nguy cơ theo hệ Lung-RADS (categories 2 → 4X) dựa trên đường kính nốt, tính điểm nguy cơ tổng hợp (combined risk) theo công thức kết hợp 60% kích thước cộng 40% điểm AI ác tính, và xuất báo cáo phân tích minh bạch giúp bác sĩ X-quang nâng cao tính nhất quán trong quy trình tầm soát ung thư phổi đại trà.

---

## 3. Đối Tượng và Phạm Vi Nghiên Cứu

### 3.1 Đối Tượng Nghiên Cứu

Đối tượng nghiên cứu là ảnh CT lồng ngực chuẩn từ tập dữ liệu LIDC-IDRI, bao gồm 1.010 bệnh nhân thu thập từ nhiều cơ sở y tế khác nhau (đa trung tâm). Dữ liệu ảnh có định dạng DICOM, kích thước trong mặt phẳng 512×512 pixel, độ dày lát cắt trong khoảng 1–3 mm. Mỗi ca bệnh có annotation từ 4 bác sĩ X-quang đọc độc lập theo quy trình blind read, sau đó unblinded rereading. Đối tượng phát hiện là nốt phổi có đường kính từ 3 mm trở lên, phù hợp với định nghĩa nốt phổi trong quy ước LIDC (Armato et al., 2011).

### 3.2 Phạm Vi Nghiên Cứu

**Trong phạm vi nghiên cứu:**

- Toàn bộ tập dữ liệu LIDC-IDRI với phân chia cố định: 812 bệnh nhân huấn luyện / 99 bệnh nhân kiểm định / 99 bệnh nhân kiểm thử (tập kiểm thử được khóa sau lần đo đầu tiên, commit `cd48359`).
- Pipeline 4 tầng: phân đoạn UNet++ EfficientNet-B5 SCSE 2.5D → lọc FPR DenseNet121-3D → phân loại ác tính DenseNet121-3D.
- Đánh giá phát hiện theo chỉ số F1, FROC, CPM với quy tắc khớp nốt 15 mm cố định (quy tắc này khác với tiêu chuẩn LUNA16 chính thức và được báo cáo tường minh trong luận văn).
- So sánh thống kê giữa mô hình đề xuất và MONAI RetinaNet 3D trên cùng tập kiểm thử (Wilcoxon signed-rank test theo bệnh nhân).
- Triển khai webapp hỗ trợ đọc với 3 chế độ hiển thị và chức năng ghi nhận phán quyết lâm sàng.
- Phân loại nguy cơ Lung-RADS và tính điểm nguy cơ tổng hợp.

**Ngoài phạm vi nghiên cứu:**

- Chẩn đoán độc lập: hệ thống được định vị là công cụ hỗ trợ đọc thứ hai, không thay thế phán quyết lâm sàng của bác sĩ.
- Kiểm định trên tập dữ liệu ngoài LIDC-IDRI (không có external validation dataset).
- Đánh giá theo chuẩn LUNA16 chính thức: quy tắc khớp nốt trong nghiên cứu này sử dụng ngưỡng 15 mm cố định, không phải quy tắc `max(3mm, diameter/2)` của LUNA16 — điều này cần được ghi nhận như một giới hạn trong so sánh với kết quả LUNA16 từ tài liệu.
- Thử nghiệm lâm sàng (clinical trial) hoặc phê duyệt quản lý (regulatory approval).
- Xử lý ảnh từ các phương thức khác (X-quang phổi, MRI, PET-CT).
- Hỗ trợ định dạng DICOM ngoài CT lồng ngực tiêu chuẩn (non-thoracic CT, non-standard reconstruction).
- Suy luận thời gian thực dưới 10 giây mỗi ca bệnh (inference speed không phải tiêu chí thiết kế của nghiên cứu này).

---

## 4. Phương Pháp và Đóng Góp Dự Kiến

### 4.1 Phương Pháp Nghiên Cứu

Nghiên cứu áp dụng phương pháp học sâu có giám sát (supervised deep learning) với framework PyTorch và thư viện MONAI. Pipeline được huấn luyện tuần tự: mô hình Stage 2 được fine-tune từ Stage 1, FPR Classifier và Malignancy Classifier được huấn luyện độc lập trên các tập patch trích xuất từ kết quả Stage 2. Kỹ thuật Stochastic Weight Averaging (SWA) được áp dụng ở Stage 2 để cải thiện khả năng tổng quát hóa. Hàm mất mát Tversky (thay vì Dice chuẩn) được lựa chọn để kiểm soát cân bằng giữa false negative và false positive trên tập dữ liệu không cân bằng.

Giao thức đánh giá theo chuẩn LUNA16-style FROC với 7 điểm FP/scan chuẩn (0.125, 0.25, 0.5, 1, 2, 4, 8), sử dụng quy tắc khớp nốt 15 mm cố định. So sánh mô hình đề xuất với MONAI RetinaNet 3D được thực hiện trên cùng tập kiểm thử cố định, với kiểm định thống kê Wilcoxon signed-rank test theo đơn vị bệnh nhân (p < 0,0001).

### 4.2 Đóng Góp Dự Kiến

1. **Pipeline 4 tầng mới cho LIDC-IDRI**: Kết hợp UNet++ EfficientNet-B5 với cơ chế attention SCSE, hàm mất mát Tversky và SWA — đạt val_dice = 0,8676 ở Stage 2, F1 = 0,618 và AUC FPR = 0,9364 trên tập kiểm thử 99 bệnh nhân.

2. **So sánh có kiểm soát giữa 2.5D segmentation và 3D detection**: Cung cấp bằng chứng thực nghiệm về sự khác biệt hiệu suất giữa hai hướng tiếp cận trên cùng tập dữ liệu, cùng quy tắc khớp nốt, và kiểm định thống kê (mine F1 = 0,618 vs. MONAI F1 = 0,533, delta = +0,085).

3. **Webapp hỗ trợ đọc 3 chế độ với verdict tracking**: Công cụ hỗ trợ bác sĩ X-quang so sánh kết quả AI và ground truth trên cùng giao diện, với chức năng ghi nhận phán quyết cho từng nốt.

4. **Khung đánh giá trung thực (honest evaluation framework)**: Báo cáo tường minh các giới hạn của giao thức đánh giá — bao gồm quy tắc khớp nốt 15 mm cố định (không phải LUNA16 chính thức), không có external validation, và phân bố kích thước nốt theo tầng — thay vì trình bày kết quả theo cách có thể gây hiểu lầm khi so sánh với tài liệu.

---

## Tài Liệu Tham Khảo

- Armato, S. G., McLennan, G., Bidaut, L., et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931.

- National Lung Screening Trial Research Team. (2011). Reduced lung-cancer mortality with low-dose computed tomographic screening. *New England Journal of Medicine*, 365(5), 395–409.

- Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2019). UNet++: A nested U-net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3–11.

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.

- Roy, A. G., Navab, N., & Wachinger, C. (2018). Concurrent spatial and channel 'squeeze & excitation' in fully convolutional networks. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 421–429.

- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4700–4708.

- Setio, A. A. A., Traverso, A., de Bel, T., et al. (2017). Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. *Medical Image Analysis*, 42, 1–13.
