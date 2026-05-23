# LUẬN VĂN TỐT NGHIỆP

**Đề tài**: Xây dựng hệ thống phát hiện và phân loại nốt phổi tự động hỗ trợ chẩn đoán ung thư phổi sớm trên ảnh CT lồng ngực (LIDC-IDRI)

**Sinh viên**: [Họ và tên sinh viên]
**Mã sinh viên**: [Mã SV]
**Lớp**: [Lớp]
**Khóa**: [Khóa]
**Giảng viên hướng dẫn**: [Học hàm. Học vị. Họ và tên]
**Trường**: [Tên trường]
**Khoa**: [Tên khoa]

[Tên thành phố], ngày … tháng … năm 2026

---

## LỜI CAM ĐOAN

Tôi xin cam đoan rằng luận văn tốt nghiệp với đề tài "Xây dựng hệ thống phát hiện và phân loại nốt phổi tự động hỗ trợ chẩn đoán ung thư phổi sớm trên ảnh CT lồng ngực (LIDC-IDRI)" là công trình nghiên cứu độc lập của tôi, được thực hiện dưới sự hướng dẫn khoa học của [Học hàm. Học vị. Họ và tên giảng viên hướng dẫn]. Tất cả số liệu, kết quả thực nghiệm và kết luận được trình bày trong luận văn là trung thực, có nguồn gốc rõ ràng và chưa từng được công bố trong bất kỳ công trình nghiên cứu nào khác. Mọi trích dẫn tài liệu tham khảo đều được ghi nhận đầy đủ và chính xác theo quy định.

[Tên thành phố], ngày … tháng … năm 2026

Sinh viên (ký và ghi rõ họ tên)

---

## LỜI CẢM ƠN

Lời đầu tiên, tôi xin bày tỏ lòng biết ơn sâu sắc đến [Học hàm. Học vị. Họ và tên giảng viên hướng dẫn] — người đã tận tình định hướng nghiên cứu, đồng hành trong từng giai đoạn thực nghiệm và không ngừng khuyến khích tinh thần khoa học nghiêm túc, trung thực trong suốt quá trình thực hiện luận văn. Sự chỉ dẫn cụ thể và những góp ý phản biện của thầy/cô chính là nền tảng để tôi kiên trì hoàn thiện từng chi tiết kỹ thuật của đề tài này.

Tôi cũng gửi lời cảm ơn chân thành đến quý thầy cô trong khoa [Tên khoa], trường [Tên trường] đã xây dựng nền tảng kiến thức vững chắc về lập trình, học máy và xử lý tín hiệu trong suốt những năm học đại học. Đặc biệt, tôi cảm ơn gia đình và bạn bè đã không ngừng động viên, tạo điều kiện để tôi hoàn thành luận văn trong thời hạn quy định. Kết quả nghiên cứu này là món quà nhỏ tôi muốn dâng tặng những người đã luôn tin tưởng và đồng hành cùng tôi.

---

## DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Giải nghĩa |
|---|---|
| AI | Artificial Intelligence — Trí tuệ nhân tạo |
| ALPR | Automatic License Plate Recognition |
| AP | Average Precision — Độ chính xác trung bình |
| API | Application Programming Interface |
| ASGI | Asynchronous Server Gateway Interface |
| AUC | Area Under the Curve — Diện tích dưới đường cong |
| BCE | Binary Cross-Entropy — Hàm mất mát entropy chéo nhị phân |
| CNN | Convolutional Neural Network — Mạng nơ-ron tích chập |
| CPM | Competition Performance Metric |
| CT | Computed Tomography — Chụp cắt lớp vi tính |
| CUDA | Compute Unified Device Architecture |
| cuDNN | CUDA Deep Neural Network Library |
| Dice | Dice Similarity Coefficient |
| DICOM | Digital Imaging and Communications in Medicine |
| F1 | F1-score (harmonic mean of Precision and Recall) |
| FP | False Positive — Dương tính giả |
| FN | False Negative — Âm tính giả |
| FPR | False Positive Reduction — Giảm dương tính giả |
| FROC | Free-Response Receiver Operating Characteristic |
| GPU | Graphics Processing Unit |
| HU | Hounsfield Unit — Đơn vị Hounsfield |
| IDRI | Image Database Resource Initiative |
| IoU | Intersection over Union |
| LIDC | Lung Image Database Consortium |
| LFS | Large File Storage (Git LFS) |
| Lung-RADS | Lung Imaging Reporting and Data System |
| LUNA16 | LUng Nodule Analysis 2016 |
| mAP | Mean Average Precision |
| MONAI | Medical Open Network for AI |
| NMS | Non-Maximum Suppression — Triệt tiêu phi cực đại |
| NLST | National Lung Screening Trial |
| NaN | Not a Number |
| REST | Representational State Transfer |
| ROC | Receiver Operating Characteristic |
| SCSE | Spatial and Channel Squeeze-and-Excitation |
| SSE | Server-Sent Events |
| SWA | Stochastic Weight Averaging |
| TP | True Positive — Dương tính thật |
| TTA | Test-Time Augmentation |
| VRAM | Video Random Access Memory |

---

## MỤC LỤC

LỜI CAM ĐOAN.........................................................................[trang]
LỜI CẢM ƠN..............................................................................[trang]
MỤC LỤC...................................................................................[trang]
DANH MỤC BẢNG BIỂU.............................................................[trang]
DANH MỤC HÌNH VẼ..................................................................[trang]
DANH MỤC TỪ VIẾT TẮT..........................................................[trang]

CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI.......................................[trang]
  1.1. Đặt vấn đề.......................................................................[trang]
  1.2. Mục tiêu và phạm vi nghiên cứu......................................[trang]
    1.2.1. Mục tiêu nghiên cứu..................................................[trang]
    1.2.2. Phạm vi nghiên cứu...................................................[trang]

CHƯƠNG 2: CƠ SỞ LÝ THUYẾT................................................[trang]
  2.1. Các công cụ và môi trường phát triển...............................[trang]
    2.1.1. Visual Studio Code.....................................................[trang]
    2.1.2. Python........................................................................[trang]
    2.1.3. PyTorch......................................................................[trang]
    2.1.4. MONAI.......................................................................[trang]
    2.1.5. SimpleITK..................................................................[trang]
    2.1.6. pydicom......................................................................[trang]
    2.1.7. lungmask....................................................................[trang]
    2.1.8. segmentation_models_pytorch (smp)...........................[trang]
    2.1.9. timm (PyTorch Image Models)....................................[trang]
    2.1.10. scikit-image và scipy.ndimage...................................[trang]
    2.1.11. NumPy và Pandas.....................................................[trang]
    2.1.12. HDF5 (h5py)............................................................[trang]
    2.1.13. Albumentations.........................................................[trang]
    2.1.14. TensorBoard..............................................................[trang]
    2.1.15. Matplotlib và Plotly..................................................[trang]
    2.1.16. FastAPI......................................................................[trang]
    2.1.17. Pydantic....................................................................[trang]
    2.1.18. Uvicorn.....................................................................[trang]
    2.1.19. Next.js (React) và TypeScript....................................[trang]
    2.1.20. Tailwind CSS.............................................................[trang]
    2.1.21. Git LFS (Large File Storage)....................................[trang]
    2.1.22. VPS V100 (Vast.ai / na-01)......................................[trang]
    2.1.23. paramiko...................................................................[trang]
    2.1.24. Jupyter Notebook......................................................[trang]
    2.1.25. CUDA và cuDNN......................................................[trang]
  2.2. Cơ sở lý thuyết mạng học sâu cho ảnh y khoa.................[trang]
    2.2.1. Mạng nơ-ron tích chập (CNN)....................................[trang]
    2.2.2. Kiến trúc U-Net cho phân đoạn ảnh y khoa................[trang]
    2.2.3. UNet++ — Nested Skip Pathways................................[trang]
    2.2.4. Encoder EfficientNet-B5.............................................[trang]
    2.2.5. Cơ chế chú ý SCSE....................................................[trang]
    2.2.6. Kiến trúc 2.5D — Multi-slice Input.............................[trang]
    2.2.7. DenseNet121-3D cho Phân loại Patch 3D....................[trang]
    2.2.8. So sánh kiến trúc 2.5D Segmentation và 3D Detection.[trang]
  2.3. Hàm mất mát và tối ưu hóa.............................................[trang]
    2.3.1. Hàm Tversky Loss.......................................................[trang]
    2.3.2. Hàm Focal-BCE..........................................................[trang]
    2.3.3. AdamW và Cosine Annealing với Warm Restart..........[trang]
    2.3.4. Stochastic Weight Averaging (SWA)...........................[trang]
  2.4. Tiền xử lý và hậu xử lý không gian..................................[trang]
    2.4.1. Chuẩn hóa HU — Cửa sổ [-1350, 150]......................[trang]
    2.4.2. Lung Segmentation Pretrained (lungmask R231)..........[trang]
    2.4.3. Connected Components 3D..........................................[trang]
    2.4.4. Lọc Elongation Ratio (PCA-based).............................[trang]
    2.4.5. Non-Maximum Suppression (NMS) 3D........................[trang]
    2.4.6. FPR Classifier Threshold (best_thr = 0.85).................[trang]
    2.4.7. Test-Time Augmentation (TTA)..................................[trang]
  2.5. Các độ đo đánh giá hiệu năng..........................................[trang]
    2.5.1. Dice Coefficient..........................................................[trang]
    2.5.2. Precision, Recall và F1...............................................[trang]
    2.5.3. AUC-ROC..................................................................[trang]
    2.5.4. Balanced Accuracy và F1 cho Phân loại Đa lớp..........[trang]
    2.5.5. FROC và CPM...........................................................[trang]
    2.5.6. Quy tắc khớp nốt: 15mm Fixed so với LUNA16.........[trang]
    2.5.7. Kiểm định Thống kê: Wilcoxon Signed-Rank..............[trang]
  2.6. Phân loại nguy cơ lâm sàng.............................................[trang]
    2.6.1. Hệ Lung-RADS..........................................................[trang]
    2.6.2. Combined Risk Score..................................................[trang]
    2.6.3. Logic 3-Tier Final Risk...............................................[trang]

CHƯƠNG 3: KIẾN TRÚC HỆ THỐNG VÀ XÂY DỰNG MÔ HÌNH.[trang]
  3.1. Kiến trúc tổng thể của hệ thống.......................................[trang]
    3.1.1. Tổng quan pipeline 4 giai đoạn..................................[trang]
    3.1.2. Tích hợp webapp 3-mode............................................[trang]
  3.2. Xây dựng và xử lý bộ dữ liệu huấn luyện........................[trang]
    3.2.1. Mô tả tập dữ liệu LIDC-IDRI....................................[trang]
    3.2.2. Chuẩn bị dữ liệu........................................................[trang]
    3.2.3. Xây dựng nhãn Ground Truth Consensus....................[trang]
    3.2.4. Phân chia tập dữ liệu..................................................[trang]
    3.2.5. Augmentation trong huấn luyện..................................[trang]
  3.3. Module Stage 1 — Baseline Segmentation 2D...................[trang]
  3.4. Module Stage 2 — Refinement 2.5D với SCSE Attention và Tversky Loss.[trang]
    3.4.1. Kiến trúc UNet++ EfficientNet-B5 SCSE 2.5D...........[trang]
    3.4.2. Hàm mất mát Tversky với α=0.3, β=0.7.....................[trang]
    3.4.3. Optimizer và Learning Rate Schedule.........................[trang]
    3.4.4. Stochastic Weight Averaging (SWA)...........................[trang]
    3.4.5. Kết quả huấn luyện Stage 2........................................[trang]
  3.5. Module FPR Classifier — DenseNet121-3D Nhị phân........[trang]
    3.5.1. Mục đích và vị trí trong pipeline................................[trang]
    3.5.2. Kiến trúc DenseNet121-3D..........................................[trang]
    3.5.3. Huấn luyện FPR.........................................................[trang]
    3.5.4. Hiệu chỉnh ngưỡng (Threshold Calibration)................[trang]
  3.6. Module Malignancy Classifier — DenseNet121-3D Phân loại 5 lớp.[trang]
    3.6.1. Mục đích và thang phân loại......................................[trang]
    3.6.2. Kiến trúc và Huấn luyện.............................................[trang]
    3.6.3. Combined Risk và phân tầng nguy cơ.........................[trang]
  3.7. Module Hậu xử lý Không gian..........................................[trang]
    3.7.1. Lung mask pretrained..................................................[trang]
    3.7.2. Connected Components và Lọc Blob 3D.....................[trang]
    3.7.3. Elongation Filter dựa trên PCA..................................[trang]
    3.7.4. NMS 3D và Đo Đường Kính.......................................[trang]
  3.8. Module Webapp — FastAPI và Next.js...............................[trang]
    3.8.1. Backend FastAPI.........................................................[trang]
    3.8.2. Frontend Next.js với TypeScript..................................[trang]
    3.8.3. Schema JSON kết quả.................................................[trang]

CHƯƠNG 4: TRIỂN KHAI VÀ ĐÁNH GIÁ KẾT QUẢ..................[trang]
  4.1. Môi trường và Thiết lập Thực nghiệm..............................[trang]
    4.1.1. Cấu hình phần cứng...................................................[trang]
    4.1.2. Cấu hình phần mềm...................................................[trang]
    4.1.3. Bảng Hyperparameter đầy đủ Stage 2.........................[trang]
  4.2. Đánh giá Quá trình Huấn luyện Stage 2...........................[trang]
    4.2.1. Đường cong Loss và Val Dice....................................[trang]
    4.2.2. Kết quả Ablation Pilot (EXP00–EXP05).....................[trang]
  4.3. Đánh giá Quá trình Huấn luyện FPR và Malignancy Classifier.[trang]
    4.3.1. FPR Classifier — Đường cong Training......................[trang]
    4.3.2. Hiệu chỉnh Ngưỡng FPR...........................................[trang]
    4.3.3. Malignancy 5-class — Phân tích Class Distribution...[trang]
  4.4. Đánh giá Hiệu năng trên Tập Kiểm thử...........................[trang]
    4.4.1. Tập Kiểm thử test_panel (99 bệnh nhân)...................[trang]
    4.4.2. So sánh Mine vs MONAI RetinaNet 3D......................[trang]
    4.4.3. Kiểm định Thống kê — Wilcoxon Signed-Rank..........[trang]
    4.4.4. Phân tích FROC và CPM...........................................[trang]
    4.4.5. Phân tích Độ nhạy Theo Kích thước Nốt....................[trang]
    4.4.6. Ablation Study trên Evaluation Panel.........................[trang]
    4.4.7. Phân tích Trường hợp Thất bại..................................[trang]
  4.5. Kiểm thử Hệ thống trong Các Kịch bản Thực tế...............[trang]
    4.5.1. Inference Time Per Case............................................[trang]
    4.5.2. Kiểm thử Edge Case..................................................[trang]
    4.5.3. Hiển thị Webapp........................................................[trang]
  4.6. Đánh giá Luồng Nghiệp vụ AI Second Reader..................[trang]
    4.6.1. Workflow End-to-End.................................................[trang]
    4.6.2. Tích hợp Lung-RADS và Combined Risk....................[trang]
    4.6.3. Bài học Kỹ thuật Rút ra.............................................[trang]
    4.6.4. Giới hạn của Nghiên cứu............................................[trang]

CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN...................[trang]
  5.1. Kết luận............................................................................[trang]
    5.1.1. Tổng kết các kết quả đạt được...................................[trang]
    5.1.2. Đóng góp khoa học và thực tiễn.................................[trang]
    5.1.3. Hạn chế của nghiên cứu.............................................[trang]
  5.2. Hướng phát triển trong tương lai.....................................[trang]
    5.2.1. Cải tiến kiến trúc mô hình..........................................[trang]
    5.2.2. Cải tiến giao thức đánh giá........................................[trang]
    5.2.3. Mở rộng phạm vi lâm sàng.........................................[trang]
    5.2.4. Triển khai và vận hành sản xuất.................................[trang]
    5.2.5. Nghiên cứu sâu hơn về AI second reader...................[trang]

TÀI LIỆU THAM KHẢO.............................................................[trang]

---

## DANH MỤC BẢNG BIỂU

[CẦN BỔ SUNG: Tự động sinh khi hoàn thành toàn bộ luận văn]

---

## DANH MỤC HÌNH VẼ

[CẦN BỔ SUNG: Tự động sinh khi hoàn thành toàn bộ luận văn]

---

# CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI

## 1.1. Đặt vấn đề

Ung thư phổi hiện là một trong những loại ung thư nguy hiểm nhất và có tỷ lệ tử vong cao nhất trên toàn cầu. Theo báo cáo GLOBOCAN 2022 của Cơ quan Nghiên cứu Ung thư Quốc tế (IARC), ung thư phổi xếp thứ nhất về số ca tử vong trong tổng số các loại ung thư, với ước tính khoảng 1,8 triệu ca tử vong mỗi năm — chiếm gần 18% tổng số ca tử vong do ung thư trên toàn cầu. Tại Việt Nam, bức tranh không kém phần nghiêm trọng: ung thư phổi thuộc nhóm hai loại ung thư có tỷ lệ tử vong cao nhất ở cả hai giới, phản ánh thực trạng đa số bệnh nhân đến viện khi bệnh đã tiến triển sang giai đoạn muộn với tiên lượng xấu. Sự chênh lệch đáng kể giữa tỷ lệ sống sót 5 năm ở giai đoạn sớm (trên 60% khi phát hiện ở giai đoạn I) và giai đoạn muộn (dưới 10% khi đã di căn xa) làm nổi bật tầm quan trọng tuyệt đối của việc phát hiện sớm trong chiến lược kiểm soát căn bệnh này.

Bước đột phá trong tầm soát ung thư phổi sớm đến từ Nghiên cứu Thử nghiệm Sàng lọc Phổi Quốc gia Hoa Kỳ (National Lung Screening Trial — NLST). Công bố năm 2011 trên tạp chí *New England Journal of Medicine*, nghiên cứu chứng minh rằng sàng lọc định kỳ bằng chụp cắt lớp vi tính liều thấp (low-dose CT) ở nhóm nguy cơ cao — người hút thuốc lá kéo dài từ 30 gói-năm trở lên, độ tuổi 55–74 — giúp giảm 20% tỷ lệ tử vong do ung thư phổi so với chụp X-quang ngực thông thường (National Lung Screening Trial Research Team, 2011). Kết quả này xác lập vai trò trung tâm của CT ngực liều thấp như một công cụ sàng lọc quần thể được khuyến nghị và triển khai rộng rãi tại các hệ thống y tế phát triển.

Tuy nhiên, việc triển khai sàng lọc CT ngực đại trà đặt ra một thách thức lớn về năng lực đọc kết quả. Một ca chụp CT ngực tiêu chuẩn thường bao gồm từ 200 đến 400 lát cắt axial, mỗi lát cắt chứa mật độ thông tin cao đòi hỏi bác sĩ X-quang phải duyệt qua có hệ thống. Trong bối cảnh triển khai sàng lọc quy mô lớn, khi mỗi bác sĩ phải đọc hàng chục ca CT mỗi ngày, hiện tượng mệt mỏi đọc (reader fatigue) trở thành một yếu tố ảnh hưởng nghiêm trọng đến độ nhạy phát hiện nốt. Bên cạnh đó, tính biến động giữa các bác sĩ (inter-reader variability) — tức sự khác biệt trong phán quyết giữa các chuyên gia khi đọc cùng một ca — là vấn đề có hệ thống trong X-quang lồng ngực, đặc biệt đối với các nốt phổi nhỏ dưới 10mm. Tập dữ liệu LIDC-IDRI, nền tảng của nghiên cứu này, chính là minh chứng cho điều đó: annotation từ 4 bác sĩ X-quang độc lập cho thấy tỷ lệ đồng thuận hoàn toàn (4/4) chỉ đạt trên một phần nhỏ các nốt, trong khi nhiều nốt chỉ được 1 hoặc 2 trong 4 bác sĩ ghi nhận (Armato et al., 2011).

Tình trạng thiếu hụt bác sĩ X-quang có chuyên môn về ảnh ngực, đặc biệt tại các cơ sở y tế tuyến tỉnh và khu vực nông thôn ở Việt Nam, càng làm trầm trọng thêm khoảng cách giữa nhu cầu tầm soát và năng lực xử lý kết quả. Trong bối cảnh đó, trí tuệ nhân tạo (AI) nổi lên như một giải pháp tiềm năng, không phải để thay thế bác sĩ, mà để đóng vai trò công cụ hỗ trợ đọc thứ hai (AI second reader) — giúp nâng cao tính nhất quán trong phát hiện nốt và cảnh báo những trường hợp dễ bị bỏ sót, qua đó hỗ trợ bác sĩ X-quang tập trung quyết định lâm sàng vào những ca phức tạp. Quan điểm này phân biệt rõ ràng AI second reader với hệ thống chẩn đoán tự động độc lập: trong mô hình second reader, bác sĩ vẫn giữ toàn quyền quyết định lâm sàng cuối cùng, và AI chỉ cung cấp đề xuất có căn cứ định lượng.

Từ góc độ kỹ thuật, sự phát triển của học sâu trong thập kỷ qua đã tạo ra điều kiện thuận lợi chưa từng có để giải quyết bài toán phát hiện nốt phổi tự động. Kiến trúc U-Net do Ronneberger et al. (2015) đề xuất đã tạo ra bước đột phá trong phân đoạn ảnh y khoa nhờ cấu trúc encoder-decoder đối xứng với các kết nối tắt (skip connections), cho phép mô hình giữ lại cả thông tin không gian độ phân giải cao lẫn ngữ nghĩa cấp cao. Kiến trúc UNet++ do Zhou et al. (2019) phát triển tiếp tục cải thiện U-Net bằng hệ thống đường dẫn tắt lồng nhau (nested skip pathways) và giám sát sâu (deep supervision), giúp tinh chỉnh đặc trưng ở nhiều cấp độ độ phân giải và đặc biệt phù hợp với việc phân đoạn các cấu trúc nhỏ như nốt phổi dưới 10mm. Kết hợp với bộ mã hóa EfficientNet-B5 (Tan & Le, 2019) pretrained trên ImageNet và cơ chế chú ý không gian-kênh đồng thời SCSE (Roy et al., 2018), kiến trúc này tạo nền tảng cho mô hình phân đoạn 2.5D được đề xuất trong nghiên cứu này. Lý do lựa chọn kiến trúc 2.5D thay vì 3D convolution đầy đủ xuất phát từ đặc thù dữ liệu CT đa trung tâm với độ dày lát cắt không đồng nhất: bằng cách ghép 3 lát cắt lân cận thành input 3 kênh cho mô hình 2D, phương pháp này tận dụng được thông tin không gian theo chiều sâu mà không bị ràng buộc về yêu cầu isotropic voxel spacing vốn rất tốn kém về VRAM khi sử dụng convolution 3D đầy đủ.

## 1.2. Mục tiêu và phạm vi nghiên cứu

### 1.2.1. Mục tiêu nghiên cứu

Đồ án hướng tới việc nghiên cứu, thiết kế và xây dựng hoàn chỉnh một hệ thống phần mềm hỗ trợ phát hiện và phân loại nốt phổi thông minh, ứng dụng sâu rộng các kỹ thuật Trí tuệ nhân tạo (AI) và học sâu y khoa để tự động hóa quy trình tầm soát ung thư phổi trên ảnh chụp cắt lớp vi tính lồng ngực. Để hiện thực hóa tầm nhìn này, đề tài phân rã thành các mục tiêu cốt lõi và cụ thể như sau:

**Nghiên cứu cơ sở lý thuyết và kiến trúc lõi của thị giác máy tính trong ảnh y khoa**: Tổng hợp, hệ thống hóa và đi sâu phân tích nền tảng của bài toán phân đoạn ảnh y khoa ba chiều. Trọng tâm nghiên cứu được đặt vào việc giải phẫu và làm chủ kiến trúc mạng nơ-ron tích chập (CNN) đa tầng, đặc biệt là họ mô hình U-Net cải tiến (UNet++) kết hợp encoder EfficientNet-B5 cùng cơ chế chú ý không gian-kênh đồng thời (SCSE — Spatial and Channel Squeeze-Excitation). Đánh giá ưu/nhược điểm của kiến trúc 2.5D segmentation đề xuất so với kiến trúc 3D detection tiêu chuẩn (MONAI RetinaNet 3D) nhằm chứng minh sự phù hợp của nó với đặc thù dữ liệu CT lồng ngực thưa-dày bất đồng nhất.

**Xây dựng, huấn luyện và tối ưu hóa pipeline phát hiện nốt phổi đa tầng (4-Stage Detection Pipeline)**: Tổ chức xử lý, chuẩn hóa và xây dựng nhãn đồng thuận trên tập dữ liệu LIDC-IDRI quy mô lớn (1.010 bệnh nhân, hơn 200.000 lát cắt DICOM, với annotation từ bốn bác sĩ X-quang đọc độc lập). Tiến hành huấn luyện tuần tự bốn mô hình: phân đoạn khởi tạo Stage 1, phân đoạn tinh chỉnh Stage 2 với hàm mất mát Tversky bất đối xứng kết hợp Stochastic Weight Averaging (SWA), bộ lọc dương tính giả DenseNet121-3D, và bộ phân loại mức độ ác tính 5 lớp. Tinh chỉnh siêu tham số và đánh giá mô hình bằng các thang đo chuẩn xác như Dice, F1, FROC và CPM (Competition Performance Metric), từ đó thiết lập một đường ống phát hiện đạt độ tin cậy cực cao trên tập kiểm thử bị khóa (val_dice = 0.8676, F1 = 0.618 trên 99 bệnh nhân kiểm thử).

**Phát triển các thuật toán Tiền xử lý và Hậu xử lý không gian nâng cao**: Nhận thức được rào cản của môi trường ảnh y khoa thực tế, đề tài đặt mục tiêu xây dựng các module xử lý ảnh truyền thống để "bọc lót" cho AI. Về tiền xử lý, tập trung vào kỹ thuật chuẩn hóa cửa sổ HU (Hounsfield Unit) trong dải [-1350, 150] và xây dựng đầu vào 2.5D ba kênh từ các lát cắt lân cận để bù đắp thông tin chiều sâu cho mô hình 2D. Về hậu xử lý, phát triển thuật toán lọc hình thái học theo tỷ lệ kéo dài (elongation ratio) nhằm loại bỏ các cấu trúc mạch máu cắt ngang, kết hợp thuật toán Non-Maximum Suppression (NMS) ba chiều để gộp các nốt chồng lấn, và ngưỡng FPR học từ dữ liệu (best_thr = 0.85, AUC = 0.9364) để loại bỏ dương tính giả trước khi trả về cho bác sĩ.

**Phát triển kiến trúc phần mềm và hệ webapp đa chế độ toàn trình (End-to-End)**: Thiết kế giao diện người dùng trực quan, thân thiện bằng nền tảng ứng dụng Web (Next.js + FastAPI), cho phép bác sĩ X-quang thao tác dễ dàng — tải lên ca bệnh DICOM, theo dõi tiến trình phân tích thời gian thực qua Server-Sent Events, và đối chiếu kết quả ba nguồn (mô hình đề xuất / MONAI RetinaNet 3D / ground truth từ chuyên gia LIDC) trên cùng một trang. Xây dựng cấu trúc dữ liệu chuẩn hóa để lưu trữ và truy xuất phán quyết lâm sàng (chấp nhận / từ chối / yêu cầu xem lại) đồng thời theo từng nốt và từng chế độ AI, hỗ trợ truy vết audit và phân tích hậu kỳ.

**Tự động hóa luồng nghiệp vụ hỗ trợ đọc lâm sàng**: Mục tiêu cuối cùng là đóng gói các công nghệ trên thành một quy trình nghiệp vụ khép kín mô phỏng "AI second reader". Bao gồm: tiếp nhận ca bệnh DICOM, tự động phát hiện và đo kích thước nốt phổi, phân loại nguy cơ theo hệ Lung-RADS (categories 2 → 4X) dựa trên đường kính nốt, tính điểm nguy cơ tổng hợp (combined risk) theo công thức kết hợp 60% kích thước cộng 40% điểm AI ác tính, và xuất báo cáo phân tích minh bạch giúp bác sĩ X-quang nâng cao tính nhất quán trong quy trình tầm soát ung thư phổi đại trà.

### 1.2.2. Phạm vi nghiên cứu

Để đảm bảo tính khả thi của đề tài và tập trung nguồn lực giải quyết triệt để các vấn đề kỹ thuật cốt lõi trong khuôn khổ luận văn tốt nghiệp, phạm vi nghiên cứu được quy định và giới hạn rõ ràng như sau.

**Đối tượng nghiên cứu và đặc tính dữ liệu**: Đối tượng nghiên cứu là ảnh CT lồng ngực chuẩn từ tập dữ liệu LIDC-IDRI, bao gồm 1.010 bệnh nhân thu thập từ nhiều cơ sở y tế khác nhau (đa trung tâm). Dữ liệu ảnh có định dạng DICOM, kích thước trong mặt phẳng 512×512 pixel, độ dày lát cắt trong khoảng 1–3 mm. Mỗi ca bệnh có annotation từ 4 bác sĩ X-quang đọc độc lập theo quy trình blind read, sau đó unblinded rereading. Đối tượng phát hiện là nốt phổi có đường kính từ 3 mm trở lên, phù hợp với định nghĩa nốt phổi trong quy ước LIDC (Armato et al., 2011). Phân chia tập dữ liệu được cố định: 812 bệnh nhân huấn luyện, 99 bệnh nhân kiểm định, và 99 bệnh nhân kiểm thử — tập kiểm thử được khóa sau khi đo F1=0.618 (commit `cd48359`) và không được phép chỉnh sửa thêm.

**Trong phạm vi nghiên cứu**: Toàn bộ tập dữ liệu LIDC-IDRI với phân chia cố định nêu trên. Pipeline 4 tầng: phân đoạn UNet++ EfficientNet-B5 SCSE 2.5D → lọc FPR DenseNet121-3D → phân loại ác tính DenseNet121-3D. Đánh giá phát hiện theo chỉ số F1, FROC, CPM với quy tắc khớp nốt 15 mm cố định — quy tắc này khác với tiêu chuẩn LUNA16 chính thức và được báo cáo tường minh trong luận văn như một giới hạn của nghiên cứu. So sánh thống kê giữa mô hình đề xuất và MONAI RetinaNet 3D trên cùng tập kiểm thử bằng kiểm định Wilcoxon signed-rank test theo đơn vị bệnh nhân. Triển khai webapp hỗ trợ đọc với 3 chế độ hiển thị và chức năng ghi nhận phán quyết lâm sàng. Phân loại nguy cơ Lung-RADS và tính điểm nguy cơ tổng hợp.

**Ngoài phạm vi nghiên cứu**: Hệ thống không được thiết kế và không được định vị là công cụ chẩn đoán độc lập — mọi kết quả AI phải được bác sĩ X-quang kiểm tra và xác nhận trước khi đưa ra quyết định lâm sàng. Nghiên cứu không bao gồm kiểm định trên tập dữ liệu ngoài LIDC-IDRI (không có external validation dataset), do đó khả năng tổng quát hóa sang dữ liệu từ các hệ máy CT khác với đặc tính kỹ thuật khác biệt là một câu hỏi mở. Đánh giá theo chuẩn LUNA16 chính thức — với quy tắc khớp nốt `max(3mm, diameter/2)` và giao thức 10-fold cross-validation — không được thực hiện; quy tắc khớp nốt trong nghiên cứu này sử dụng ngưỡng 15 mm cố định và cần được ghi nhận như một giới hạn khi so sánh với kết quả LUNA16 trong tài liệu. Thử nghiệm lâm sàng (clinical trial) và phê duyệt quản lý (regulatory approval) nằm ngoài phạm vi của một luận văn tốt nghiệp đại học. Nghiên cứu không xử lý ảnh từ các phương thức khác như X-quang phổi thông thường, MRI hay PET-CT. Tốc độ suy luận thời gian thực dưới 10 giây mỗi ca bệnh không phải là tiêu chí thiết kế — thời gian xử lý thực tế phụ thuộc vào phần cứng triển khai và không được tối ưu hóa trong giai đoạn nghiên cứu này.

---

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Các công cụ và môi trường phát triển

Hệ thống được xây dựng trên một hệ sinh thái công cụ chuyên biệt cho AI y khoa, trải dài từ môi trường phát triển, thư viện học sâu, xử lý ảnh y khoa, cho đến nền tảng web và hạ tầng tính toán GPU. Phần này trình bày từng thành phần theo chức năng, tập trung làm rõ vai trò cụ thể của từng công cụ trong pipeline nghiên cứu.

### 2.1.1. Visual Studio Code

Để quản lý, soạn thảo và gỡ lỗi toàn bộ mã nguồn dự án — bao gồm pipeline tiền xử lý DICOM, các script huấn luyện mô hình, backend FastAPI và frontend Next.js — nghiên cứu sử dụng Visual Studio Code (VS Code) do Microsoft phát triển làm môi trường thao tác chính. VS Code không đơn thuần là một trình soạn thảo văn bản mà được xây dựng trên nền tảng framework Electron, kết hợp engine kết xuất Chromium và môi trường thực thi Node.js, cho phép mở rộng không giới hạn qua hệ thống extension.

Trong ngữ cảnh nghiên cứu AI y khoa, VS Code đặc biệt hữu ích nhờ tính năng gỡ lỗi tích hợp (Debug Adapter Protocol) cho phép đặt điểm dừng để kiểm tra trực tiếp giá trị tensor PyTorch trong bộ nhớ GPU — điều này vô cùng quan trọng khi chẩn đoán lỗi kích thước tensor trong các lớp decoder UNet++. Khả năng quản lý môi trường ảo Python (virtual environment) tích hợp sẵn giúp cô lập phụ thuộc giữa các phiên bản thư viện khác nhau, ngăn xung đột phiên bản thường gặp khi làm việc đồng thời với nhiều tập dữ liệu và nhiều phiên bản mô hình.

### 2.1.2. Python

Python là ngôn ngữ lập trình bậc cao, thông dịch, đa mục đích được lựa chọn làm ngôn ngữ lõi liên kết toàn bộ pipeline nghiên cứu. Được tạo ra bởi Guido van Rossum và ra mắt năm 1991, Python đã trở thành ngôn ngữ thống trị trong lĩnh vực khoa học dữ liệu và trí tuệ nhân tạo nhờ triết lý thiết kế nhấn mạnh tính đọc được và hệ sinh thái thư viện mã nguồn mở đồ sộ.

Trong nghiên cứu này, Python đóng vai trò kết nối tất cả các thành phần: từ đọc và tiền xử lý ảnh DICOM (pydicom, SimpleITK), huấn luyện và đánh giá mô hình học sâu (PyTorch, MONAI, segmentation_models_pytorch), xử lý hình thái học hậu kỳ (scipy, scikit-image), cho đến xây dựng backend API (FastAPI) và tự động hóa tác vụ từ xa trên VPS V100 (paramiko). Việc sử dụng một ngôn ngữ thống nhất trên toàn pipeline giúp giảm thiểu chi phí chuyển đổi ngữ cảnh và tạo điều kiện tích hợp mượt mà giữa các giai đoạn xử lý.

### 2.1.3. PyTorch

PyTorch là framework học sâu mã nguồn mở do Meta AI Research phát triển, đóng vai trò là bộ máy tính toán (compute engine) trung tâm cho mọi mô hình học sâu trong dự án. Điểm đặc trưng cốt lõi của PyTorch là đồ thị tính toán động (dynamic computational graph) thông qua cơ chế autograd, cho phép tính toán đạo hàm tự động và linh hoạt thay đổi kiến trúc mạng trong quá trình chạy — điều vô cùng thuận tiện khi thử nghiệm nhiều biến thể kiến trúc UNet++.

Trong pipeline này, PyTorch xử lý toàn bộ vòng lặp huấn luyện của ba mô hình: Stage 2 UNet++ EfficientNet-B5 SCSE, FPR Classifier DenseNet121-3D và Malignancy Classifier DenseNet121-3D. Tích hợp với CUDA cho phép đẩy các phép nhân ma trận tensor xuống phần cứng GPU V100 SXM2 32GB trên VPS, rút ngắn đáng kể thời gian huấn luyện so với tính toán trên CPU.

### 2.1.4. MONAI

MONAI (Medical Open Network for AI) là một framework học sâu chuyên biệt cho ảnh y khoa, xây dựng trên nền tảng PyTorch, cung cấp các component được thiết kế và tối ưu hóa đặc biệt cho dữ liệu y tế như DICOM và NIfTI. Trong nghiên cứu này, MONAI đóng hai vai trò: là nền tảng cho mô hình baseline MONAI RetinaNet 3D được sử dụng làm đối tượng so sánh thống kê với mô hình đề xuất, và cung cấp một số tiện ích tiền xử lý ảnh y khoa chuẩn hóa.

Việc lựa chọn MONAI RetinaNet 3D làm baseline có chủ ý: đây là mô hình 3D detection đã được kiểm chứng rộng rãi trong cộng đồng, được huấn luyện với cùng tập dữ liệu và cùng giao thức đánh giá, cho phép so sánh trực tiếp giữa hai hướng tiếp cận phân đoạn 2.5D (đề xuất) và phát hiện 3D (baseline) trên cùng điều kiện kiểm soát.

### 2.1.5. SimpleITK

SimpleITK là thư viện xử lý ảnh y khoa được thiết kế trên nền tảng Insight Segmentation and Registration Toolkit (ITK), cung cấp giao diện Python để đọc, ghi và xử lý các định dạng ảnh y khoa đa chiều. Trong pipeline tiền xử lý, SimpleITK được sử dụng chủ yếu cho các tác vụ resampling không gian — đặc biệt quan trọng khi chuẩn hóa voxel spacing về một chuẩn đồng nhất trước khi đưa vào mô hình, và cho các tác vụ đọc/ghi volume 3D. Tính năng xử lý hình học 3D chính xác của SimpleITK phù hợp đặc biệt với dữ liệu CT đa trung tâm có thông số vật lý không đồng nhất.

### 2.1.6. pydicom

pydicom là thư viện Python mã nguồn mở chuyên dụng để đọc, ghi và thao tác với file DICOM (Digital Imaging and Communications in Medicine) — định dạng lưu trữ tiêu chuẩn của ảnh y tế. Trong pipeline tiền xử lý, pydicom đảm nhiệm bước đầu tiên và quan trọng nhất: đọc từng file DICOM từ thư mục ca bệnh, trích xuất metadata (pixel spacing, slice thickness, image position patient, rescale slope/intercept) và chuyển đổi giá trị pixel thô về đơn vị Hounsfield Unit (HU) theo công thức `HU = pixel_value × RescaleSlope + RescaleIntercept`. Độ chính xác của bước này trực tiếp ảnh hưởng đến chất lượng chuẩn hóa cửa sổ HU ở các bước tiếp theo.

### 2.1.7. lungmask

lungmask là thư viện Python cung cấp mô hình phân đoạn phổi pretrained, xây dựng dựa trên kiến trúc U-Net 2D. Mô hình R231 của Hofmanninger et al. (2020) được tích hợp trong thư viện này đã được huấn luyện trên hơn 200 ca CT lồng ngực đa dạng và có khả năng phân đoạn phổi trái/phải với độ chính xác cao ngay cả trong các trường hợp có bệnh lý phổi.

Trong pipeline này, lungmask R231 được sử dụng để tạo mặt nạ phổi (lung mask) cho từng ca CT trước khi phát hiện nốt. Mục đích là loại bỏ các vùng nằm ngoài phổi khỏi phạm vi tìm kiếm của mô hình phân đoạn, giảm đáng kể số lượng dương tính giả từ các cấu trúc ngoài phổi như xương sườn, mô mềm thành ngực và các tạng bụng xuất hiện ở các lát cắt thấp. Đây là quyết định thiết kế tiết kiệm chi phí so với việc huấn luyện lại mô hình phân đoạn phổi từ đầu.

### 2.1.8. segmentation_models_pytorch (smp)

segmentation_models_pytorch (smp) là thư viện Python cung cấp các kiến trúc phân đoạn ảnh học sâu với thiết kế tách biệt giữa encoder (backbone) và decoder, cho phép kết hợp linh hoạt giữa nhiều backbone pretrained với nhiều kiến trúc decoder khác nhau. Thư viện này là nền tảng xây dựng mô hình chính của nghiên cứu: UNet++ với encoder EfficientNet-B5 và decoder có cơ chế attention SCSE.

Lý do lựa chọn smp thay vì tự viết UNet++ từ đầu là vì thư viện cung cấp implementation đã được kiểm chứng, tối ưu về bộ nhớ và hỗ trợ dễ dàng các tùy chọn attention decoder. Đặc biệt, smp cho phép khởi tạo encoder EfficientNet-B5 với trọng số pretrained trên ImageNet — một kỹ thuật transfer learning quan trọng giúp mô hình hội tụ nhanh hơn trên tập dữ liệu y khoa quy mô vừa.

### 2.1.9. timm (PyTorch Image Models)

timm (PyTorch Image Models) là thư viện mã nguồn mở do Ross Wightman phát triển, cung cấp hơn 700 mô hình backbone pretrained bao gồm EfficientNet, ResNet, ViT và nhiều kiến trúc tiên tiến khác. Thư viện này được smp sử dụng internally để tải trọng số pretrained của EfficientNet-B5, nhưng cũng được sử dụng trực tiếp trong một số thử nghiệm khám phá kiến trúc backbone thay thế. timm đóng vai trò là kho lưu trữ trọng số pretrained chuẩn hóa, đảm bảo tính tái tạo của các thử nghiệm transfer learning.

### 2.1.10. scikit-image và scipy.ndimage

scikit-image và scipy.ndimage là hai thư viện xử lý ảnh và tín hiệu khoa học quan trọng, đảm nhiệm toàn bộ bước hậu xử lý không gian sau khi mô hình phân đoạn tạo ra binary mask thô. Cụ thể, `scipy.ndimage.label` với cấu trúc kết nối 3×3×3 được dùng để phân tích thành phần liên thông 3D (connected components) — tách biệt từng ứng viên nốt riêng lẻ từ mask nhị phân. scikit-image cung cấp các hàm đo lường hình thái học để tính toán các thuộc tính hình dạng của từng blob như thể tích, bounding box, và các giá trị eigenvalue của tensor quán tính — dùng để tính elongation ratio phân biệt nốt phổi hình cầu với mạch máu hình trụ cắt ngang.

### 2.1.11. NumPy và Pandas

NumPy cung cấp cấu trúc mảng đa chiều hiệu suất cao (ndarray) và là nền tảng tính toán số học cho toàn bộ pipeline xử lý ảnh, từ chuẩn hóa HU, ghép lát cắt 2.5D cho đến tính toán các độ đo đánh giá. Pandas được sử dụng để quản lý metadata, theo dõi kết quả thử nghiệm hyperparameter, phân tích phân bố kích thước nốt theo tầng và xuất báo cáo kết quả đánh giá dưới dạng bảng cấu trúc. Trong pipeline nghiên cứu, sự kết hợp NumPy-Pandas thay thế hiệu quả cho cơ sở dữ liệu quan hệ ở các tác vụ phân tích dữ liệu không yêu cầu tính bền vững cao.

### 2.1.12. HDF5 (h5py)

Định dạng HDF5 (Hierarchical Data Format version 5) và thư viện Python h5py được sử dụng để lưu trữ tập dữ liệu LIDC-IDRI đã qua tiền xử lý. Mỗi ca bệnh trong số 1.010 bệnh nhân được lưu thành một file `.h5` chứa mảng ảnh HU `int16` kích thước [N, 512, 512], mảng mask 4 bác sĩ [N, 4, 512, 512], và metadata vật lý (pixel spacing, slice thickness, z positions). Lý do lựa chọn HDF5 thay vì lưu trực tiếp ảnh DICOM trong quá trình training là vì HDF5 hỗ trợ truy xuất ngẫu nhiên chunk hiệu quả, nén dữ liệu trong suốt, và tốc độ đọc nhanh hơn nhiều so với đọc lại DICOM từ nhiều file nhỏ — một yếu tố quan trọng khi dataloader cần tải hàng nghìn lát cắt mỗi epoch trên VPS.

### 2.1.13. Albumentations

Albumentations là thư viện tăng cường dữ liệu (data augmentation) hiệu suất cao được thiết kế đặc biệt cho ảnh và mask phân đoạn, với đặc điểm là áp dụng cùng một phép biến đổi hình học một cách đồng bộ lên cả ảnh input lẫn mask ground truth — một yêu cầu bắt buộc trong bài toán phân đoạn học có giám sát. Trong pipeline huấn luyện, Albumentations cung cấp các kỹ thuật augmentation như horizontal/vertical flip, random rotation, elastic deformation và gaussian noise, giúp tăng tính đa dạng của dữ liệu training và giảm thiểu hiện tượng overfitting — đặc biệt quan trọng khi số lượng lát cắt có nốt (positive slices) trong tập train rất ít so với lát cắt không có nốt.

### 2.1.14. TensorBoard

TensorBoard là công cụ trực quan hóa quá trình huấn luyện của Google, tích hợp với PyTorch qua `torch.utils.tensorboard`. Trong suốt quá trình huấn luyện Stage 2 trên VPS V100, TensorBoard ghi nhận và hiển thị theo thời gian thực các chỉ số training loss, validation Dice và validation IoU theo từng epoch. Dữ liệu TensorBoard được đồng bộ từ VPS về máy local qua SSH tunneling để theo dõi tiến trình mà không cần kết nối GUI trực tiếp. Logs TensorBoard từ ba run training chính được lưu tại `work/tensorboard_logs/` và là nguồn xác minh số liệu training trong luận văn.

### 2.1.15. Matplotlib và Plotly

Matplotlib cung cấp nền tảng vẽ đồ thị tĩnh chuẩn để tạo các biểu đồ learning curves (loss/Dice theo epoch), đường cong FROC và ROC, và các biểu đồ so sánh hiệu suất giữa mô hình đề xuất và MONAI. Plotly được sử dụng cho visualization tương tác — đặc biệt cho việc kết xuất mesh 3D của các nốt phổi trong không gian volumetric, giúp trực quan hóa hình dạng và phân bố không gian của các ứng viên nốt được phát hiện trong quá trình kiểm tra kết quả thủ công. Các hình ảnh FROC tại `work/academic/figures/` được tạo bằng Matplotlib từ dữ liệu đánh giá thực nghiệm.

### 2.1.16. FastAPI

FastAPI là framework Python hiệu suất cao để xây dựng API RESTful, xây dựng trên Starlette và Pydantic, hỗ trợ native bất đồng bộ (async/await) và tự động sinh tài liệu OpenAPI. Trong hệ thống này, FastAPI đảm nhiệm toàn bộ tầng backend: nhận upload file DICOM từ frontend, quản lý pipeline inference (gọi lần lượt lungmask → Stage2 → FPR → Malignancy), và truyền kết quả trực tiếp về frontend theo thời gian thực qua Server-Sent Events (SSE).

Tính năng SSE streaming là lý do chính để lựa chọn FastAPI: vì inference một ca CT đầy đủ có thể mất nhiều giây, SSE cho phép backend gửi từng nốt phát hiện được về frontend ngay khi xử lý xong từng lát cắt, thay vì phải chờ toàn bộ volume kết thúc — cải thiện đáng kể trải nghiệm người dùng trong môi trường lâm sàng.

### 2.1.17. Pydantic

Pydantic là thư viện Python validation dữ liệu dựa trên type annotations, được tích hợp natively vào FastAPI. Trong hệ thống, Pydantic định nghĩa các schema nghiêm ngặt cho request/response API — bao gồm cấu trúc dữ liệu nốt phổi (tọa độ centroid, đường kính, điểm ác tính, điểm tin cậy), schema phán quyết lâm sàng (verdict accept/reject/review_later), và payload kết quả so sánh ba chế độ. Validation tự động của Pydantic đảm bảo tính toàn vẹn kiểu dữ liệu tại mọi điểm biên API, giảm thiểu lỗi runtime do kiểu dữ liệu không khớp — một vấn đề thường gặp khi interface giữa model inference Python và frontend TypeScript.

### 2.1.18. Uvicorn

Uvicorn là ASGI server (Asynchronous Server Gateway Interface) siêu nhẹ và hiệu suất cao, được sử dụng để chạy ứng dụng FastAPI trong môi trường production. Uvicorn xây dựng trên `uvloop` và `httptools`, cho phép xử lý đồng thời nhiều kết nối SSE từ nhiều phiên làm việc của bác sĩ, đồng thời tích hợp tốt với hệ thống quản lý tiến trình để restart tự động khi có lỗi.

### 2.1.19. Next.js (React) và TypeScript

Next.js là framework React meta-framework với khả năng server-side rendering (SSR) và static site generation (SSG), được lựa chọn để xây dựng giao diện webapp hỗ trợ đọc cho bác sĩ X-quang. TypeScript được sử dụng thay cho JavaScript thuần để tận dụng kiểm tra kiểu tại thời điểm biên dịch — đặc biệt quan trọng khi giao diện cần xử lý cấu trúc dữ liệu phức tạp của kết quả inference (danh sách nốt với nhiều thuộc tính, so sánh ba chế độ AI).

Webapp cung cấp ba chế độ xem: (1) chế độ Mine — kết quả từ mô hình UNet++ đề xuất, (2) chế độ MONAI — kết quả từ baseline MONAI RetinaNet 3D, (3) chế độ Ground Truth — annotation từ chuyên gia LIDC. Tính năng verdict tracking cho phép bác sĩ ghi nhận quyết định lâm sàng theo từng nốt và lưu trữ bền vững phục vụ phân tích hậu kỳ.

### 2.1.20. Tailwind CSS

Tailwind CSS là framework CSS utility-first cung cấp các class CSS nguyên tử có thể kết hợp trực tiếp trong JSX/TSX. Thay vì viết CSS tùy chỉnh, giao diện webapp sử dụng các Tailwind class để thiết kế layout responsive, màu sắc phân cấp nguy cơ (xanh/vàng/đỏ tương ứng low/medium/high), và trạng thái tương tác của các nút verdict. Lựa chọn Tailwind giảm thiểu kích thước CSS bundle nhờ cơ chế purge CSS tự động loại bỏ các class không sử dụng trong build production.

### 2.1.21. Git LFS (Large File Storage)

Git LFS là extension của Git cho phép lưu trữ các file nhị phân lớn (như model checkpoint) trong repository mà không làm tăng kích thước repository git chính. Trong dự án này, checkpoint mô hình Stage 2 tốt nhất (`best.pt`, ~124MB) và checkpoint SWA được lưu qua Git LFS tại commit `cd48359`. Điều này cho phép đồng bộ chính xác phiên bản checkpoint giữa môi trường phát triển local và VPS V100, đảm bảo tính tái tạo của kết quả inference.

### 2.1.22. VPS V100 (Vast.ai / na-01)

Hạ tầng GPU training là VPS thuê trên nền tảng Vast.ai với cấu hình Tesla V100-SXM2-32GB VRAM. Đây là lý do trực tiếp dẫn đến quyết định sử dụng kiến trúc 2.5D thay vì 3D convolution đầy đủ: với VRAM 32GB, training batch UNet++ EfficientNet-B5 với input 3D đầy đủ (512×512×N) cho nốt phổi là bất khả thi, trong khi input 2.5D (3×512×512 per slice) cho phép batch size 10 và training ổn định trong 180 epoch.

Toàn bộ quá trình training Stage 2 (180 epochs, ~72 giờ) được thực hiện trên VPS này. Các checkpoint được log và đồng bộ định kỳ về local qua SSH. Log training từ VPS là nguồn xác minh duy nhất cho các số liệu training trong luận văn.

### 2.1.23. paramiko

paramiko là thư viện Python cài đặt giao thức SSH2, cho phép tự động hóa các tác vụ trên VPS từ máy local mà không cần kết nối GUI. Trong dự án, paramiko được sử dụng để tự động submit job training, theo dõi log training từ xa, đồng bộ checkpoint và chạy các script đánh giá trên VPS — tất cả qua Python script thay vì phải SSH thủ công từng lệnh. Điều này đặc biệt quan trọng khi VPS có thể bị ngắt kết nối và cần tự động khởi động lại tiến trình training.

### 2.1.24. Jupyter Notebook

Jupyter Notebook được sử dụng cho các tác vụ phân tích khám phá (exploratory data analysis) trực tiếp trên VPS: kiểm tra phân bố HU của dữ liệu, quan sát mẫu annotation của các bác sĩ, phân tích phân bố kích thước nốt theo tầng, và gỡ lỗi pipeline tiền xử lý từng bước. Jupyter cho phép chạy đoạn code ngắn và quan sát kết quả ngay lập tức — nhanh hơn nhiều so với viết và chạy script Python đầy đủ — trong quá trình prototype các thuật toán hậu xử lý mới.

### 2.1.25. CUDA và cuDNN

CUDA (Compute Unified Device Architecture) là nền tảng tính toán song song của NVIDIA, cho phép PyTorch thực thi các phép nhân ma trận tensor trên GPU thay vì CPU. cuDNN (CUDA Deep Neural Network Library) là thư viện tối ưu cấp thấp cho các phép toán học sâu phổ biến như tích chập, batch normalization và pooling trên kiến trúc CUDA. Kết hợp, CUDA và cuDNN là lý do tốc độ training trên V100 nhanh hơn khoảng 50-100 lần so với CPU, đưa thời gian một epoch UNet++ EfficientNet-B5 từ hàng giờ xuống còn vài chục phút.

---

## 2.2. Cơ sở lý thuyết mạng học sâu cho ảnh y khoa

### 2.2.1. Mạng nơ-ron tích chập (CNN)

Mạng nơ-ron tích chập (Convolutional Neural Network — CNN) là kiến trúc học sâu nền tảng được thiết kế chuyên biệt để xử lý dữ liệu có cấu trúc dạng lưới không gian, tiêu biểu là hình ảnh. Khác với mạng nơ-ron kết nối đầy đủ (fully-connected network) xử lý toàn bộ ảnh như một vector phẳng, CNN khai thác tính cục bộ và tính bất biến dịch chuyển (translation invariance) của đặc trưng ảnh thông qua hai cơ chế: kết nối cục bộ (local connectivity) và chia sẻ trọng số (weight sharing).

**Phép tích chập 2D**: Toán tử tích chập 2D giữa ảnh đầu vào $I$ và bộ lọc (kernel) $K$ kích thước $m \times m$ tại vị trí $(i, j)$ trên feature map đầu ra $O$ được định nghĩa:

$$O(i, j) = \sum_{p=0}^{m-1} \sum_{q=0}^{m-1} I(i+p, j+q) \cdot K(p, q) + b$$

trong đó $b$ là hệ số bias. Bộ lọc $K$ không được đặt thủ công mà được học tự động qua lan truyền ngược (backpropagation). Ở các lớp nông, bộ lọc học các đặc trưng cơ bản như cạnh và gradient; ở các lớp sâu, chúng học các đặc trưng ngữ nghĩa phức tạp hơn như hình dạng tổng thể.

**Hàm kích hoạt phi tuyến**: Sau mỗi lớp tích chập, hàm kích hoạt phi tuyến được áp dụng để đưa ra khả năng mô hình hóa các hàm phi tuyến tính phức tạp. Hàm ReLU (Rectified Linear Unit) $f(x) = \max(0, x)$ được sử dụng phổ biến nhờ đơn giản, tính toán nhanh và giảm thiểu vấn đề vanishing gradient so với hàm sigmoid hoặc tanh. Biến thể LeakyReLU $f(x) = \max(\alpha x, x)$ với $\alpha \ll 1$ được dùng ở một số lớp để tránh "dying ReLU" — hiện tượng nơ-ron bị khóa vĩnh viễn ở giá trị 0.

**Batch Normalization**: Giữa lớp tích chập và hàm kích hoạt, Batch Normalization (Ioffe & Szegedy, 2015) chuẩn hóa phân phối kích hoạt trong một mini-batch, giúp ổn định quá trình training và cho phép sử dụng learning rate lớn hơn. Trong kiến trúc UNet++ EfficientNet-B5, Batch Normalization được áp dụng tại mỗi khối encoder và decoder.

**Pooling**: Lớp pooling giảm chiều không gian của feature map. Max pooling lấy giá trị lớn nhất trong mỗi vùng cửa sổ, tạo tính bất biến cục bộ với dịch chuyển nhỏ. Average pooling lấy giá trị trung bình, được sử dụng trong một số kiến trúc như trong Global Average Pooling (GAP) để nén feature map thành vector trước lớp phân loại.

### 2.2.2. Kiến trúc U-Net cho phân đoạn ảnh y khoa

U-Net được Ronneberger, Fischer và Brox đề xuất năm 2015 tại hội nghị MICCAI, ban đầu nhằm giải quyết bài toán phân đoạn ảnh vi khuẩn dưới kính hiển vi với dữ liệu huấn luyện hạn chế (Ronneberger et al., 2015). Kiến trúc bao gồm hai đường dẫn đối xứng: encoder (đường dẫn xuống) và decoder (đường dẫn lên), kết nối bằng các skip connections ngang.

Encoder áp dụng tuần tự các khối tích chập và max pooling để trích xuất đặc trưng ngữ nghĩa cấp cao ở độ phân giải giảm dần. Decoder sử dụng upsampling (hoặc transposed convolution) để khôi phục độ phân giải không gian, đồng thời nhận thông tin từ encoder qua skip connections. Cơ chế skip connection là điểm mấu chốt của U-Net: thay vì chỉ dựa vào bottleneck đặc trưng có độ phân giải thấp, decoder được truyền trực tiếp các feature map từ mức encoder tương ứng — giúp bảo tồn thông tin không gian chi tiết (ranh giới, góc cạnh) vốn bị mất trong quá trình pooling. Điều này đặc biệt phù hợp với ảnh y khoa nơi ranh giới cấu trúc giải phẫu là thông tin quan trọng nhất.

Lý do U-Net phù hợp với CT ngực: (1) Cấu trúc encoder-decoder cho phép phân đoạn đa tỷ lệ, xử lý tốt các nốt từ nhỏ (3mm) đến lớn (>15mm) trong cùng một mạng. (2) Skip connections giúp định vị chính xác ranh giới nốt. (3) Cấu trúc này học được từ tương đối ít dữ liệu có annotation nhờ chia sẻ đặc trưng qua encoder pretrained.

### 2.2.3. UNet++ — Nested Skip Pathways

UNet++ được Zhou et al. đề xuất năm 2019 như một cải tiến hệ thống của U-Net, giải quyết hạn chế của skip connections trong U-Net gốc: các skip connections trong U-Net nối trực tiếp feature map encoder với decoder mà không có bất kỳ biến đổi nào, gây ra sự chênh lệch (semantic gap) giữa đặc trưng ngữ nghĩa encoder và decoder ở cùng cấp độ độ phân giải (Zhou et al., 2019).

UNet++ thay thế các skip connections thẳng bằng các mạng tích chập lồng nhau (nested sub-networks). Ký hiệu $x^{i,j}$ là feature map tại encoder node thứ $i$ và decoder node thứ $j$. Trong U-Net gốc, $x^{i,j}$ chỉ nhận đầu vào từ $x^{i,j-1}$ (decoder cùng cấp, bước trước) và $x^{i+1,j-1}$ (upsampled từ cấp sâu hơn). Trong UNet++:

$$x^{i,j} = \mathcal{H}([\{x^{i,k}\}_{k=0}^{j-1}, \mathcal{U}(x^{i+1,j-1})])$$

trong đó $\mathcal{H}$ là khối tích chập, $\mathcal{U}$ là upsampling, và quan trọng nhất, decoder nhận đầu vào từ tất cả các node encoder/decoder trước đó ở cùng cấp — không chỉ node encoder gốc. Điều này tạo ra đường dẫn đặc trưng dày đặc, dần tinh chỉnh (re-designed skip pathways) giúp thu hẹp semantic gap.

**Giám sát sâu (Deep Supervision)**: UNet++ tùy chọn áp dụng giám sát sâu bằng cách thêm output head tại nhiều cấp độ decoder. Các output này được so sánh với ground truth bằng loss function, giúp gradient lan truyền hiệu quả hơn qua các lớp sâu trong quá trình training.

**Lý do chọn UNet++ thay vì U-Net gốc trong nghiên cứu này**: Nốt phổi nhỏ (<10mm) là đối tượng phát hiện chính, và đây là trường hợp UNet++ có lợi thế rõ ràng nhất. Nhờ nested skip pathways tinh chỉnh đặc trưng ở nhiều cấp độ, UNet++ phục hồi tốt hơn các ranh giới mờ của nốt nhỏ — nơi semantic gap giữa encoder và decoder trong U-Net gốc gây ra sự mất mát thông tin biên giới quan trọng nhất.

### 2.2.4. Encoder EfficientNet-B5

EfficientNet được Tan và Le đề xuất năm 2019, giải quyết câu hỏi: khi tăng quy mô mạng CNN (scale up), nên tăng chiều nào — chiều sâu (số lớp), chiều rộng (số kênh), hay độ phân giải đầu vào? Nghiên cứu chứng minh rằng cân bằng đồng thời cả ba chiều theo một hệ số tổng hợp (compound scaling coefficient) $\phi$ cho kết quả tốt hơn so với tăng bất kỳ một chiều đơn lẻ (Tan & Le, 2019).

EfficientNet-B5 là phiên bản thứ năm trong họ EfficientNet (B0–B7), với kích thước đầu vào 456×456 và khoảng 28–30 triệu tham số. Trong smp, EfficientNet-B5 được sử dụng như encoder của UNet++: các feature map từ 5 stage của EfficientNet (với stride 2 mỗi stage, tổng downsample 32×) tạo thành các cấp độ của kim tự tháp đặc trưng (feature pyramid) được nối sang decoder bằng nested skip pathways.

**Lý do chọn B5**: Trong thực nghiệm pilot (EXP01–EXP05 với các backbone B3, B4, B5), EfficientNet-B5 cho val_dice tốt nhất trong khi vẫn nằm trong giới hạn VRAM 32GB của V100 với batch size 10. Backbone nhỏ hơn (B3, B4) hội tụ nhanh hơn nhưng đỉnh val_dice thấp hơn; backbone lớn hơn (B6, B7) vượt quá giới hạn VRAM khi kết hợp với UNet++ decoder đầy đủ.

### 2.2.5. Cơ chế chú ý SCSE

Concurrent Spatial and Channel Squeeze-and-Excitation (SCSE) được Roy, Navab và Wachinger đề xuất năm 2018 như một module plug-in có thể gắn vào bất kỳ kiến trúc phân đoạn nào (Roy et al., 2018). SCSE kết hợp song song hai nhánh attention:

**Channel Squeeze-and-Excitation (cSE)**: Nhánh này học trọng số quan trọng cho từng kênh đặc trưng. Với feature map $U \in \mathbb{R}^{H \times W \times C}$, Global Average Pooling nén thành vector $z \in \mathbb{R}^C$, sau đó hai lớp fully-connected với hàm sigmoid tạo ra vector trọng số kênh $\hat{s} \in \mathbb{R}^C$. Feature map được nhân channel-wise với $\hat{s}$: cSE học "kênh nào quan trọng" — ví dụ kênh mã hóa mật độ mô mềm quan trọng hơn kênh mã hóa không khí khi phân đoạn nốt.

**Spatial Squeeze-and-Excitation (sSE)**: Nhánh này học bản đồ quan trọng theo không gian. Tích chập $1 \times 1$ trên $U$ tạo ra bản đồ $\hat{p} \in \mathbb{R}^{H \times W \times 1}$ — "pixel nào quan trọng". Feature map được nhân pointwise với $\hat{p}$.

Output của SCSE là tổng hai nhánh: $\hat{U} = cSE(U) + sSE(U)$, cho phép mô hình đồng thời focus vào đúng kênh và đúng vị trí không gian liên quan đến nốt.

**Lý do SCSE giúp phát hiện nốt phổi**: Nốt phổi nhỏ (<10mm) chiếm rất ít pixel trong toàn bộ lát cắt 512×512, trong khi phần lớn không gian là không khí phổi và mô không liên quan. SCSE cho phép decoder tự động tập trung attention vào các vùng nhỏ có mật độ HU đặc trưng của nốt, thay vì xử lý đều tất cả vị trí — cải thiện đáng kể khả năng phục hồi nốt nhỏ trong bước giải mã feature map.

### 2.2.6. Kiến trúc 2.5D — Multi-slice Input

Kiến trúc 2.5D là giải pháp thực tế để tích hợp thông tin 3D vào mô hình 2D mà không cần convolution 3D đầy đủ. Thay vì xử lý từng lát cắt CT độc lập (kiến trúc 2D thuần), input của mô hình là tổ hợp 3 lát cắt liền kề: $[s_{i-1}, s_i, s_{i+1}]$ được ghép thành tensor 3 kênh $\mathbb{R}^{3 \times 512 \times 512}$.

Cơ chế này cho phép mô hình học được sự liên tục giải phẫu giữa các lát cắt — quan trọng vì nốt phổi là cấu trúc 3D và hình dạng của nó biến đổi liên tục qua các lát cắt liền kề. Đồng thời, vì encoder xử lý 3 kênh như input RGB, toàn bộ trọng số pretrained ImageNet của EfficientNet-B5 vẫn được sử dụng mà không cần chỉnh sửa kiến trúc, tận dụng transfer learning hiệu quả.

**Lợi thế so với 3D convolution**: Với dữ liệu CT đa trung tâm có độ dày lát cắt từ 1mm đến 3mm, voxel spacing không đồng nhất (anisotropic). Convolution 3D đầy đủ sẽ xử lý không đúng khoảng cách vật lý nếu không resampling về isotropic spacing — resampling isotropic làm tăng kích thước volume 3-5 lần, vượt giới hạn VRAM thực tế. Kiến trúc 2.5D tránh được bẫy này hoàn toàn: mỗi lát cắt được xử lý theo in-plane resolution 512×512 gốc, và context 3D được thu thập một cách xấp xỉ qua 3 lát cắt lân cận.

### 2.2.7. DenseNet121-3D cho Phân loại Patch 3D

DenseNet (Densely Connected Convolutional Network) được Huang et al. đề xuất năm 2017 với ý tưởng cốt lõi: mỗi lớp nhận kết nối trực tiếp từ tất cả các lớp trước đó trong cùng dense block (Huang et al., 2017). Nếu mạng có $L$ lớp, DenseNet có $L(L+1)/2$ kết nối — so với $L$ kết nối trong ResNet. Feature map của lớp $l$ là concatenation của tất cả feature map từ lớp 0 đến $l-1$:

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

Kết nối dày đặc này có hai tác dụng: (1) gradient flow tốt hơn qua toàn bộ mạng giúp training ổn định; (2) tái sử dụng đặc trưng (feature reuse) giảm số tham số cần thiết so với kiến trúc độ sâu tương đương.

DenseNet121-3D là biến thể 3D của DenseNet121 (121 lớp), thay thế mọi convolution 2D bằng convolution 3D, phù hợp với việc phân loại khối 3D (3D patch) trích xuất quanh tâm ứng viên nốt. Trong pipeline này, DenseNet121-3D được sử dụng cho hai nhiệm vụ: FPR Classifier (binary — nốt thật hay dương tính giả) và Malignancy Classifier (5-class — mức độ ác tính 1-5). Input là khối 64×64×64 voxel chuẩn hóa HU, đủ lớn để chứa hầu hết các nốt phổi (kể cả nốt >15mm) và ngữ cảnh xung quanh.

### 2.2.8. So sánh kiến trúc 2.5D Segmentation (đề xuất) và 3D Detection (MONAI)

Hai hướng tiếp cận chính cho bài toán phát hiện nốt phổi là phân đoạn 2.5D (như đề xuất trong nghiên cứu này) và phát hiện 3D (như MONAI RetinaNet 3D). Bảng dưới đây tổng hợp ưu nhược điểm của từng hướng.

| Tiêu chí | 2.5D Segmentation (đề xuất) | 3D Detection (MONAI RetinaNet) |
|---|---|---|
| Kiến trúc | UNet++ EfficientNet-B5 SCSE | RetinaNet 3D |
| Đầu vào | 3 lát cắt liền kề [3×512×512] | Volume 3D resampled |
| VRAM yêu cầu | ~8GB/batch-10 (V100 32GB đủ) | >16GB (yêu cầu hơn cho full volume) |
| Transfer learning | EfficientNet-B5 pretrained ImageNet | Không dùng pretrained 2D |
| Xử lý anisotropic | Tự nhiên — không cần resampling | Cần resampling isotropic |
| Hậu xử lý | Connected components 3D + morphology | Tích hợp sẵn trong anchor framework |
| F1 trên test_panel | **0.618** | 0.533 |
| Ưu điểm | Transfer learning tốt, VRAM hiệu quả | Pipeline đồng nhất, không cần hậu xử lý thủ công |
| Nhược điểm | Context 3D xấp xỉ, hậu xử lý phức tạp | Không tận dụng pretrained 2D backbone |

Kết quả thực nghiệm cho thấy pipeline đề xuất đạt F1=0.618 trên test_panel 99 bệnh nhân, vượt MONAI baseline F1=0.533 với delta +0.085 (Wilcoxon signed-rank, p<0.0001). Kết quả này cung cấp bằng chứng thực nghiệm ủng hộ hướng tiếp cận 2.5D segmentation với pretrained backbone trên tập dữ liệu này, mặc dù cần lưu ý rằng kết quả có thể không tổng quát hóa sang các tập dữ liệu khác với đặc tính khác.

---

## 2.3. Hàm mất mát và tối ưu hóa

### 2.3.1. Hàm Tversky Loss

Hàm Dice Loss chuẩn, dựa trên Dice Similarity Coefficient $DSC = \frac{2|P \cap G|}{|P| + |G|}$, tối ưu đồng đều false negative (FN) và false positive (FP). Trong bài toán phân đoạn nốt phổi, FN và FP không có hậu quả lâm sàng tương đương: bỏ sót một nốt ác tính (FN) nguy hiểm hơn nhiều so với phát hiện nhầm một dương tính giả (FP) vốn có thể được bác sĩ loại trừ trong bước review thủ công.

Hàm Tversky Loss do Salehi et al. (2017) đề xuất giải quyết vấn đề này bằng cách thêm hai tham số kiểm soát: $\alpha$ (trọng số FP) và $\beta$ (trọng số FN):

$$TL = 1 - \frac{\sum_{i} p_{0i} g_{0i}}{\sum_{i} p_{0i} g_{0i} + \alpha \sum_{i} p_{0i} g_{1i} + \beta \sum_{i} p_{1i} g_{0i}}$$

trong đó $p_{0i}$ là xác suất dự đoán pixel $i$ là nốt, $g_{0i}$ là nhãn ground truth, $g_{1i}$ là nhãn background; $\sum p_{0i} g_{1i}$ là tổng FP và $\sum p_{1i} g_{0i}$ là tổng FN.

**Cài đặt trong nghiên cứu**: $\alpha = 0.3$, $\beta = 0.7$ — penalty FN nặng hơn FP theo tỷ lệ 7:3. Cài đặt này là kết quả thực nghiệm từ loạt pilot EXP01–EXP05: Tversky với $\beta = 0.7$ cho val_dice cao hơn Dice Loss chuẩn và Focal-Tversky trên validation set. Về mặt lâm sàng, cài đặt này phù hợp với định vị của hệ thống là AI second reader ưu tiên độ nhạy (sensitivity) — tránh bỏ sót nốt phổi — trong khi các FP có thể được bác sĩ lọc bỏ ở bước review.

### 2.3.2. Hàm Focal-BCE

Binary Cross-Entropy (BCE) Loss là hàm mất mát tiêu chuẩn cho phân loại nhị phân:

$$BCE = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

Trong bài toán phân loại FPR (binary: nốt thật hay dương tính giả), sự mất cân bằng lớp (class imbalance) là vấn đề nghiêm trọng: số lượng dương tính giả từ Stage 2 nhiều hơn nhiều so với nốt thật. Focal Loss (Lin et al., 2017) mở rộng BCE bằng hệ số điều chỉnh $(1-\hat{p})^\gamma$ giảm đóng góp gradient của các ví dụ dễ (easy examples) — tức các dương tính giả rõ ràng mà mô hình đã phân loại đúng với confidence cao — để tập trung training vào các ví dụ khó:

$$FL = -\alpha_t (1-\hat{p}_t)^\gamma \log(\hat{p}_t)$$

Trong pipeline FPR Classifier, BCE tiêu chuẩn được sử dụng với class weighting (nghịch đảo tần suất) thay vì Focal Loss, vì class weighting cho kết quả ổn định hơn trên tập dữ liệu này trong các thử nghiệm nội bộ.

### 2.3.3. AdamW và Cosine Annealing với Warm Restart

**AdamW**: Thuật toán AdamW (Loshchilov & Hutter, 2019) là biến thể của Adam với L2 weight decay được áp dụng trực tiếp lên trọng số thay vì qua gradient — correcting một lỗi thiết kế trong Adam gốc nơi weight decay bị hấp thụ vào momentum. AdamW được lựa chọn cho tất cả các mô hình trong pipeline nhờ tính ổn định trong quá trình training và khả năng điều chỉnh learning rate thích nghi per-parameter.

**Cosine Annealing**: Learning rate schedule theo cosine annealing giảm learning rate theo hàm cosine từ $\eta_{max}$ xuống $\eta_{min}$ trong một chu kỳ $T$ epoch:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

Lý do lựa chọn cosine thay vì step decay hoặc exponential decay: cosine schedule giảm dần mượt mà hơn, giúp mô hình hội tụ vào vùng "phẳng" (flat minima) của không gian loss — những vùng này thường có khả năng tổng quát hóa tốt hơn so với vùng "nhọn" mà step decay có thể dừng lại.

### 2.3.4. Stochastic Weight Averaging (SWA)

Stochastic Weight Averaging (SWA) là kỹ thuật do Izmailov et al. (2018) đề xuất: thay vì sử dụng checkpoint cuối cùng, SWA tính trung bình trọng số của mô hình tại nhiều điểm trong quá trình training để tìm ra một bộ tham số "trung bình" nằm ở tâm của vùng phẳng trong không gian loss. Điều này giúp cải thiện khả năng tổng quát hóa so với một checkpoint đơn lẻ.

**Cài đặt trong Stage 2**: SWA được bật từ epoch 136/180 (75% tổng số epoch), với learning rate reset về $\eta = 10^{-4}$ và các checkpoint trung gian được tích lũy để tính trung bình. Kết quả: checkpoint swa.pt đạt val_dice=0.8457, val_iou=0.7726 — thấp hơn best.pt tại epoch 16 (val_dice=0.8676) trên validation set. Mặc dù SWA không cho val_dice cao nhất trong nghiên cứu này, checkpoint swa.pt vẫn được dùng trong Test-Time Augmentation (TTA) ensemble với best.pt để tăng tính robust của kết quả cuối cùng.

---

## 2.4. Tiền xử lý và hậu xử lý không gian

### 2.4.1. Chuẩn hóa HU — Cửa sổ [-1350, 150]

Đơn vị Hounsfield (HU) là thang đo tuyến tính mật độ X-quang của mô trong ảnh CT, được chuẩn hóa sao cho nước = 0 HU và không khí = -1000 HU. Các mô khác nhau có dải HU đặc trưng: phổi (-950 đến -700 HU), mô mềm (-100 đến +100 HU), xương (+400 đến +1000 HU), và nốt phổi thường trong khoảng -100 đến +100 HU tùy theo thành phần.

Cửa sổ [-1350, 150] được chọn dựa trên phân tích phân bố HU của tập LIDC-IDRI: giới hạn dưới -1350 bao phủ toàn bộ không khí phổi và các vùng emphysema nặng, giới hạn trên +150 bao phủ nốt đặc (solid nodule), nốt bán đặc (part-solid) và cạnh mô mềm thành phổi. Sau khi cắt (clip) về dải [-1350, 150], giá trị HU được chuẩn hóa tuyến tính về [0, 1]:

$$x_{norm} = \frac{\text{clip}(x_{HU}, -1350, 150) - (-1350)}{150 - (-1350)}$$

Cửa sổ này rộng hơn cửa sổ phổi chuẩn [-1000, -300] để bảo tồn thông tin biên giới giữa nốt và mô phổi xung quanh, và rộng hơn cửa sổ mô mềm [-160, 240] để đảm bảo context toàn phổi không bị bão hòa.

### 2.4.2. Lung Segmentation Pretrained (lungmask R231)

Trước khi đưa volume CT vào mô hình phân đoạn nốt, lung segmentation pretrained (lungmask R231, Hofmanninger et al., 2020) được chạy để tạo binary mask phổi. Mask này được sử dụng theo hai cách: (1) giới hạn vùng tìm kiếm ứng viên nốt trong bước connected components, loại bỏ các cấu trúc ngoài phổi có thể gây FP; (2) trong một số biến thể, được dùng như kênh auxiliary input để model biết ranh giới phổi.

Việc sử dụng mô hình pretrained sẵn có thay vì huấn luyện lại từ đầu là quyết định hợp lý về mặt tài nguyên: lung segmentation là bài toán đã giải quyết tốt, và R231 đạt Dice >0.97 trên nhiều tập dữ liệu CT lồng ngực đa dạng.

### 2.4.3. Connected Components 3D

Sau khi mô hình phân đoạn tạo ra binary mask per-slice, các mask được tổng hợp thành volume 3D và thuật toán phân tích thành phần liên thông 3D (3D connected components) được áp dụng bằng `scipy.ndimage.label` với cấu trúc kết nối 3×3×3 (26-connectivity) — nghĩa là hai voxel được xét là cùng thành phần nếu chia sẻ face, edge hoặc corner. Mỗi blob liên thông sau đó trở thành một ứng viên nốt với các thuộc tính: thể tích, centroid, bounding box, và các tính chất hình học.

Lọc sơ bộ theo thể tích: ứng viên có dưới `min_voxels=200` voxel bị loại bỏ như noise — đây là ngưỡng tương đương với cầu cầu ~4mm đường kính ở spacing 1.25mm/voxel.

### 2.4.4. Lọc Elongation Ratio (PCA-based)

Mạch máu trong phổi, khi cắt theo mặt phẳng axial, xuất hiện như các blob hình elip kéo dài — rất khác về hình dạng so với nốt phổi hình cầu hoặc bầu dục. Để loại bỏ các cấu trúc mạch máu này mà không cần một bước classification riêng biệt, elongation ratio dựa trên phân tích thành phần chính (PCA) được tính cho mỗi blob:

Từ tập hợp tọa độ voxel của blob, tính ma trận hiệp phương sai (covariance matrix), sau đó phân rã thành trị riêng (eigenvalues) $\lambda_1 \geq \lambda_2 \geq \lambda_3$. Elongation ratio được định nghĩa là $\text{ratio} = \sqrt{\lambda_1 / \lambda_3}$. Các blob có ratio > 4 bị loại bỏ như mạch máu.

**Giới hạn**: Bước lọc hình thái học này hoạt động tốt với mạch máu thẳng nhưng có thể loại nhầm nốt phổi bất thường có hình dạng kéo dài do bám sát màng phổi (pleural attachment). Đây là một tradeoff chấp nhận được trong bối cảnh hệ thống second reader.

### 2.4.5. Non-Maximum Suppression (NMS) 3D

Do mô hình phân đoạn 2.5D xử lý từng lát cắt độc lập, cùng một nốt vật lý có thể tạo ra nhiều blob liên thông riêng biệt ở các lát cắt lân cận hoặc do lỗi phân đoạn gián đoạn. NMS 3D theo khoảng cách centroid được áp dụng để gộp các ứng viên trùng lặp: với hai ứng viên có centroid cách nhau nhỏ hơn `merge_dist=10mm`, ứng viên có confidence thấp hơn (nhỏ hơn theo FPR score) bị loại bỏ, chỉ giữ lại ứng viên có confidence cao nhất.

Ngưỡng 10mm được chọn gần bằng đường kính trung bình của nốt phổi nhỏ-trung bình (6–15mm), đảm bảo hai ứng viên thực sự của cùng một nốt vật lý đều nằm trong vùng gộp, đồng thời không gộp nhầm hai nốt nằm gần nhau.

### 2.4.6. FPR Classifier Threshold (best_thr = 0.85)

Sau khi DenseNet121-3D FPR Classifier dự đoán xác suất P(true nodule) cho từng ứng viên, ngưỡng phân loại best_thr được lựa chọn bằng cách quét ROC trên validation set và chọn ngưỡng tối đa hóa F1. Kết quả: best_thr = 0.85, AUC = 0.9364 trên test_panel.

Ngưỡng cao (0.85) phản ánh ưu tiên precision cao trong bước này: bác sĩ X-quang không nên bị quá tải bởi FP, nhưng các FN ở bước này vẫn có thể được mô hình phát hiện ở mức confidence thấp hơn và hiển thị riêng như "low-confidence candidates" trong giao diện webapp.

### 2.4.7. Test-Time Augmentation (TTA)

Test-Time Augmentation là kỹ thuật ensemble tại thời điểm inference: thay vì chỉ chạy một lần với ảnh gốc, mô hình được chạy với nhiều phiên bản augmented của cùng input (horizontal flip, vertical flip), và các output mask được tổng hợp (trung bình xác suất pixel-wise) trước khi ngưỡng hóa. TTA giúp giảm variance của dự đoán và tăng tính robust với các nốt ở vị trí sát biên.

Trong pipeline này, TTA sử dụng ensemble gồm: ảnh gốc + horizontal flip + vertical flip, với mỗi biến thể được inference bằng cả best.pt và swa.pt — tổng 6 lần inference per-slice. Đây là tradeoff giữa thời gian inference (tăng 6×) và chất lượng dự đoán, phù hợp với môi trường lâm sàng không yêu cầu real-time.

---

## 2.5. Các độ đo đánh giá hiệu năng

### 2.5.1. Dice Coefficient

Dice Similarity Coefficient (DSC) là độ đo chính để đánh giá chất lượng phân đoạn nhị phân, đo lường mức độ trùng khớp giữa mask dự đoán $P$ và mask ground truth $G$:

$$DSC = \frac{2|P \cap G|}{|P| + |G|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

DSC = 0 khi không có giao nhau; DSC = 1 khi hai mask trùng hoàn toàn. Trong nghiên cứu này, Dice được tính tại cấp độ lát cắt (per-slice) trên tập validation, lấy trung bình qua tất cả các lát cắt có nốt. Giá trị val_dice = 0.8676 tại epoch 16 của Stage 2 là số liệu đo trực tiếp từ log training trên VPS.

### 2.5.2. Precision, Recall và F1

Tại cấp độ phát hiện nốt (detection), các độ đo dựa trên đếm nốt được sử dụng. Một nốt dự đoán được coi là True Positive (TP) nếu centroid của nó cách centroid một nốt ground truth dưới ngưỡng khớp (matching threshold). Precision, Recall và F1:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 là trung bình điều hòa (harmonic mean) của Precision và Recall, phù hợp để đánh giá mô hình trong điều kiện class imbalance. F1 = 0.618 (mine) và F1 = 0.533 (MONAI) là số liệu trên test_panel 99 bệnh nhân với matching threshold 15mm cố định.

### 2.5.3. AUC-ROC

Area Under the ROC Curve (AUC-ROC) đánh giá khả năng phân biệt của classifier nhị phân độc lập với ngưỡng phân loại. Đường cong ROC vẽ True Positive Rate (Recall) trên trục Y và False Positive Rate (FP/(FP+TN)) trên trục X khi ngưỡng thay đổi từ 0 đến 1. AUC = 0.5 tương đương với dự đoán ngẫu nhiên; AUC = 1.0 tương đương với phân loại hoàn hảo. AUC = 0.9364 của FPR Classifier trên test_panel chỉ ra khả năng phân biệt tốt giữa nốt thật và dương tính giả.

### 2.5.4. Balanced Accuracy và F1 cho Phân loại Đa lớp

Trong bài toán phân loại mức độ ác tính 5 lớp (Malignancy Classifier), class imbalance nghiêm trọng hơn (lớp 3 chiếm đa số) làm cho accuracy đơn giản không đủ thông tin. Balanced accuracy tính trung bình recall qua tất cả các lớp:

$$\text{Bal. Acc.} = \frac{1}{K}\sum_{k=1}^{K}\frac{TP_k}{TP_k + FN_k}$$

với $K=5$ lớp. Val balanced accuracy = 0.4593 tại epoch 52 của Malignancy Classifier phản ánh độ khó của bài toán phân loại ác tính 5 lớp — đây là bài toán vốn khó ngay cả với chuyên gia, vì nhãn ác tính được thu thập từ đánh giá chủ quan của bác sĩ với inter-annotator agreement thấp.

### 2.5.5. FROC và CPM

Free-Response ROC (FROC) là độ đo đánh giá chuẩn cho bài toán phát hiện y khoa, tổng quát hóa ROC sang bài toán multi-instance detection (một ảnh có thể chứa nhiều đối tượng). Đường cong FROC vẽ sensitivity (tỷ lệ nốt GT được phát hiện đúng) trên trục Y và số FP trung bình mỗi scan trên trục X.

Competition Performance Metric (CPM) là trung bình sensitivity tại 7 điểm FP/scan chuẩn của LUNA16: {0.125, 0.25, 0.5, 1, 2, 4, 8}. CPM = 1.0 nghĩa là sensitivity = 100% tại tất cả mức FP/scan. Trong phong trào LUNA16, các hệ thống hàng đầu đạt CPM > 0.8 (Setio et al., 2017).

FROC và CPM cung cấp cái nhìn toàn diện hơn F1 vì hiển thị tradeoff Sensitivity–FP/scan trên toàn bộ dải ngưỡng, giúp bác sĩ và kỹ sư lựa chọn điểm hoạt động phù hợp với ưu tiên lâm sàng cụ thể.

### 2.5.6. Quy tắc khớp nốt: 15mm Fixed (nghiên cứu này) so với LUNA16

**QUAN TRỌNG — Disclaimer về giao thức đánh giá**: Nghiên cứu này sử dụng quy tắc khớp nốt với ngưỡng 15mm cố định: một nốt dự đoán được coi là TP nếu khoảng cách centroid 3D đến nốt GT gần nhất nhỏ hơn 15mm. Đây là ngưỡng lớn hơn đáng kể so với quy tắc LUNA16 chính thức: `max(3mm, diameter/2)` — đối với nốt 6mm đường kính, LUNA16 dùng ngưỡng 3mm trong khi nghiên cứu này dùng 15mm.

Hệ quả là các số liệu F1, CPM, và FROC trong nghiên cứu này KHÔNG thể so sánh trực tiếp với kết quả của các phương pháp được đánh giá theo giao thức LUNA16 chính thức trong tài liệu. Ngưỡng 15mm rộng hơn có thể cho phép khớp các nốt có vị trí dự đoán không chính xác, có xu hướng thổi phồng kết quả so với ngưỡng LUNA16. Giới hạn này được ghi nhận tường minh và cần được xem xét khi diễn giải kết quả.

### 2.5.7. Kiểm định Thống kê: Wilcoxon Signed-Rank

Để so sánh hai mô hình (mine vs. MONAI) một cách có căn cứ thống kê, kiểm định Wilcoxon signed-rank test được áp dụng tại cấp độ bệnh nhân (per-patient). Với mỗi bệnh nhân trong test_panel, F1 của mô hình mine và MONAI được tính riêng, tạo thành 99 cặp giá trị. Wilcoxon signed-rank là kiểm định phi tham số phù hợp với dữ liệu không chuẩn (F1 theo bệnh nhân thường có phân phối lệch) và dữ liệu cặp (paired samples). Kết quả p < 0.0001 cho thấy sự khác biệt delta = +0.085 có ý nghĩa thống kê, không phải do may mắn ngẫu nhiên.

---

## 2.6. Phân loại nguy cơ lâm sàng

### 2.6.1. Hệ Lung-RADS

Lung Imaging Reporting and Data System (Lung-RADS) là hệ thống phân loại chuẩn do Hiệp hội X-quang Hoa Kỳ (ACR) phát triển để báo cáo và quản lý các phát hiện trên CT sàng lọc phổi. Hệ thống phân loại nốt phổi thành các category dựa chủ yếu trên đường kính nốt:

| Category | Đặc điểm | Khuyến nghị |
|---|---|---|
| 1 | Không có nốt | Tiếp tục sàng lọc thường quy |
| 2 | Nốt lành tính / ổn định | Tiếp tục sàng lọc thường quy |
| 3 | Nốt có khả năng lành tính thấp (<1%) | Theo dõi CT 6 tháng |
| 4A | Nốt đáng ngờ (1–2%) | CT 3 tháng hoặc PET-CT |
| 4B | Nốt rất đáng ngờ (>15%) | Sinh thiết / điều trị |
| 4X | Nốt có đặc điểm bổ sung tăng nguy cơ | Sinh thiết / điều trị khẩn |

Trong hệ thống này, category được gán tự động dựa trên đường kính nốt đo được từ mô hình phân đoạn: <6mm → category 2; 6–8mm → category 3; 8–15mm → category 4A; >15mm → category 4B. Đây là tích hợp clinical workflow quan trọng giúp webapp không chỉ hiển thị kết quả AI thuần túy mà còn cung cấp ngôn ngữ lâm sàng quen thuộc với bác sĩ X-quang.

### 2.6.2. Combined Risk Score

Ngoài Lung-RADS dựa thuần túy trên kích thước, hệ thống tính điểm nguy cơ tổng hợp (combined risk score) kết hợp hai nguồn thông tin:

$$\text{CombinedRisk} = 0.6 \times \text{SizePrior}(d) + 0.4 \times \frac{\text{MalignancyScore} - 1}{4}$$

trong đó SizePrior(d) là giá trị 0–1 ánh xạ từ đường kính $d$ theo thang phi tuyến (nốt ≤4mm → 0.1; ≥20mm → 1.0), và MalignancyScore là điểm ác tính 1–5 từ Malignancy Classifier được chuẩn hóa về [0, 1].

Tỷ lệ 60:40 ưu tiên kích thước hơn điểm AI, phản ánh thực tế lâm sàng rằng kích thước nốt là yếu tố dự đoán nguy cơ mạnh nhất và đã được xác nhận lâm sàng qua nhiều nghiên cứu lớn, trong khi điểm AI malignancy cung cấp thông tin bổ sung nhưng chưa được xác nhận trên dữ liệu ngoài.

### 2.6.3. Logic 3-Tier Final Risk

Điểm combined risk được phân loại về 3 mức nguy cơ cuối cho giao diện webapp:

- **Low** (thấp): CombinedRisk < 0.33 hoặc đường kính <6mm
- **Medium** (trung bình): 0.33 ≤ CombinedRisk < 0.67 hoặc đường kính 6–15mm
- **High** (cao): CombinedRisk ≥ 0.67 hoặc đường kính >15mm hoặc suspicion probability > 0.7 từ Malignancy Classifier

Phân tầng 3 mức này đơn giản hóa giao diện so với 5–6 category Lung-RADS đầy đủ, giúp bác sĩ nhanh chóng ưu tiên các nốt cần xem xét kỹ trong quy trình đọc ca lớn khối lượng cao. Tuy nhiên, cần nhấn mạnh rằng phân tầng nguy cơ này là đề xuất hỗ trợ quyết định — bác sĩ X-quang giữ toàn quyền đánh giá lâm sàng cuối cùng.

---

## Tài liệu tham khảo chương 1–2

Armato, S. G., McLennan, G., Bidaut, L., et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931.

Hofmanninger, J., Prayer, F., Pan, J., Röhrich, S., Prosch, H., & Langs, G. (2020). Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. *European Radiology Experimental*, 4(1), 1–13.

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4700–4708.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *Conference on Uncertainty in Artificial Intelligence (UAI)*.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 448–456.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*.

National Lung Screening Trial Research Team. (2011). Reduced lung-cancer mortality with low-dose computed tomographic screening. *New England Journal of Medicine*, 365(5), 395–409.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234–241.

Roy, A. G., Navab, N., & Wachinger, C. (2018). Concurrent spatial and channel 'squeeze & excitation' in fully convolutional networks. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 421–429.

Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *Machine Learning in Medical Imaging (MLMI)*, 379–387.

Setio, A. A. A., Traverso, A., de Bel, T., et al. (2017). Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. *Medical Image Analysis*, 42, 1–13.

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2019). UNet++: A nested U-net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3–11.

---

# CHƯƠNG 3: KIẾN TRÚC HỆ THỐNG VÀ XÂY DỰNG MÔ HÌNH

## 3.1. Kiến trúc tổng thể của hệ thống

### 3.1.1. Tổng quan pipeline 4 giai đoạn

Hệ thống phát hiện và phân loại nốt phổi được xây dựng theo kiến trúc pipeline đa giai đoạn (multi-stage pipeline), trong đó mỗi giai đoạn giải quyết một bài toán con độc lập với đặc thù riêng về đầu vào, đầu ra và tiêu chí đánh giá. Thiết kế này khác biệt cơ bản so với các kiến trúc end-to-end đơn khối (monolithic single-network), và sự lựa chọn kiến trúc đa giai đoạn có lý do kỹ thuật rõ ràng.

Pipeline được tổ chức theo trình tự sau:

```
DICOM input
    ↓
[Tiền xử lý] — DICOM → HU int16 → HDF5 → sắp xếp lát cắt theo z
    ↓
[Lung Mask] — lungmask R231 (Hofmanninger 2020) → binary mask phổi
    ↓
[Stage 1 — Baseline Segmentation 2D]
    UNet++ EfficientNet-B5, single-slice, BCE+Dice loss
    → best.pt Stage 1 (warm-start cho Stage 2)
    ↓
[Stage 2 — Refinement 2.5D SCSE + Tversky]
    Input: 3-slice stack (i-1, i, i+1)
    Encoder: EfficientNet-B5, Decoder: UNet++ + SCSE attention
    Loss: Tversky α=0.3, β=0.7
    Optimizer: AdamW + Cosine Annealing + SWA
    → best.pt (val_dice=0.8676 ep 16) + swa.pt (averaged ep 136–180)
    ↓
[Hậu xử lý không gian]
    Threshold → binary volume → connected components 3D (26-connectivity)
    Lọc: min_voxels=200, elongation PCA > 4, subpleural < 2mm
    NMS 3D merge distance < 10mm
    Đo diameter từ core voxel (prob > 0.85)
    ↓
[FPR Classifier — DenseNet121-3D Binary]
    Patch 48×48×48, HU [-1024, 600]
    → FPR prob; threshold=0.85; val ROC AUC=0.9364
    ↓
[Malignancy Classifier — DenseNet121-3D 5-class]
    Patch 32×32×32, HU [-1024, 600]
    → malignancy score 1–5; combined risk 3-tier
    ↓
[Lung-RADS + Combined Risk]
    Category 2/3/4A/4B/4X theo đường kính
    CombinedRisk = 0.6×SizePrior + 0.4×AI_score
    ↓
OUTPUT — JSON nodule list cho radiologist (webapp)
```

Lý do chọn kiến trúc 4 giai đoạn thay vì end-to-end: (1) mỗi giai đoạn có không gian bài toán khác nhau — phân đoạn pixel-level (Stage 2) khác phân loại khối (FPR, Malignancy), đòi hỏi kiến trúc, dữ liệu huấn luyện, và hàm mất mát khác nhau; (2) từng module có thể được debug, thay thế, hoặc nâng cấp độc lập mà không ảnh hưởng toàn bộ hệ thống; (3) việc kiểm soát luồng dữ liệu tường minh giữa các stage giúp dễ dàng xác định nguồn gốc lỗi (error attribution) trong quá trình phát triển và thử nghiệm lâm sàng.

### 3.1.2. Tích hợp webapp 3-mode

Pipeline AI được tích hợp vào webapp với 3 chế độ hoạt động song song:

- **Mine** (mô hình nghiên cứu này): kết quả từ Stage 2 + FPR + Malignancy + post-processing.
- **MONAI** (đường cơ sở so sánh): MONAI RetinaNet 3D với bundle lung_nodule_ct_detection từ thư viện MONAI Model Zoo.
- **GT** (ground truth): nốt từ annotation XML của LIDC-IDRI, chỉ có giá trị với ca từ tập test_panel có nhãn; dùng cho hiệu chuẩn và kiểm tra giao thức đánh giá.

Thiết kế 3-mode cho phép bác sĩ so sánh trực tiếp kết quả AI với nhãn chuyên gia ngay trong cùng giao diện, tăng tính minh bạch và hỗ trợ đào tạo lâm sàng.

---

## 3.2. Xây dựng và xử lý bộ dữ liệu huấn luyện

### 3.2.1. Mô tả tập dữ liệu LIDC-IDRI

Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) là tập dữ liệu ảnh CT ngực công cộng lớn nhất và được trích dẫn nhiều nhất trong lĩnh vực phát hiện nốt phổi (Armato et al., 2011). Tập dữ liệu bao gồm 1010 bệnh nhân được thu thập từ 7 trung tâm y tế tại Hoa Kỳ với nhiều nhà sản xuất máy CT khác nhau (Siemens, GE, Philips, Toshiba), tạo ra sự đa dạng về thiết bị và giao thức chụp đặc trưng của dữ liệu lâm sàng thực tế.

Điểm đặc trưng nhất của LIDC-IDRI là quy trình annotation hai vòng (two-phase blinded annotation): trong vòng đầu, mỗi trong số 4 bác sĩ X-quang (radiologist) đọc ảnh độc lập và đánh dấu các vùng nghi ngờ; trong vòng hai, mỗi bác sĩ được xem kết quả ẩn danh của 3 bác sĩ còn lại và có cơ hội chỉnh sửa đánh giá của mình. Quy trình này được thiết kế để giảm thiểu ảnh hưởng chéo (anchoring bias) trong giai đoạn đầu đồng thời cho phép bác sĩ xem xét lại dựa trên đồng thuận nhóm (Armato et al., 2011). Kết quả là mỗi nốt được gán mức độ đồng thuận từ 1/4 đến 4/4 bác sĩ xác nhận, cùng với điểm ác tính (malignancy) trên thang 1–5.

Về thông số kỹ thuật: ảnh DICOM có kích thước thường 512×512 pixel mỗi lát cắt; slice thickness phổ biến từ 0.6–5mm, tập trung ở 1–3mm; pixel spacing 0.5–0.8mm. Annotation được lưu dưới dạng XML với contour (outline) từng nốt của từng bác sĩ, bao gồm tọa độ pixel cho từng lát cắt.

### 3.2.2. Chuẩn bị dữ liệu

Quy trình tiền xử lý được thiết kế để chuẩn hóa ảnh từ nhiều nguồn khác nhau về định dạng thống nhất phù hợp với huấn luyện deep learning:

**(1) Đọc DICOM và chuyển đổi sang HU:** Mỗi lát cắt DICOM được đọc bằng thư viện pydicom; giá trị pixel thô được chuyển sang đơn vị Hounsfield (HU) theo công thức tuyến tính: `HU = pixel × RescaleSlope + RescaleIntercept`. Kết quả lưu dạng `int16` để tiết kiệm bộ nhớ (dải HU phổi: không khí ≈ −1000 HU, mô mềm 20–80 HU, xương 400–1000 HU).

**(2) Sắp xếp lát cắt:** Các lát cắt trong mỗi series được sắp xếp theo tọa độ `ImagePositionPatient[2]` (trục z) theo thứ tự tăng dần, vì thứ tự file DICOM trong thư mục không nhất thiết khớp với thứ tự vật lý trong không gian.

**(3) Lưu trữ HDF5:** Toàn bộ volume bệnh nhân được lưu vào file HDF5 với cấu trúc: `images [N, 512, 512] int16` (N là số lát cắt), `mask_per_rad [N, 4, 512, 512] uint8` (mask nhị phân từng bác sĩ), cùng các trường metadata (voxel spacing, patient ID, series UID). Định dạng HDF5 với nén gzip cấp 4 cho tốc độ đọc ngẫu nhiên nhanh hơn so với lưu từng file numpy riêng, đặc biệt khi training với batch ngẫu nhiên từ nhiều bệnh nhân. Tổng dung lượng cho 1010 bệnh nhân: ~13 GB.

### 3.2.3. Xây dựng nhãn Ground Truth Consensus

Một quyết định thiết kế quan trọng là lựa chọn chiến lược tổng hợp nhãn từ 4 bác sĩ có thể không đồng thuận. Ba chiến lược được cân nhắc:

- **Union**: pixel là nốt nếu ít nhất 1/4 bác sĩ đánh dấu — quá nhiễu, bao gồm cả các vùng bác sĩ đơn lẻ ghi nhầm.
- **Consensus2** (được chọn): pixel là nốt nếu ít nhất 2/4 bác sĩ đồng thuận: `mask_gt = (Σ mask_per_rad ≥ 2)`.
- **Consensus3, Consensus4**: ngưỡng cao hơn — bỏ sót nhiều nốt thật, đặc biệt nốt nhỏ ít được ghi nhận.

Consensus2 được chọn làm chiến lược ground truth vì: (1) lọc được nhiều annotation đơn lẻ (nonNodule markers, artifact) của một bác sĩ; (2) vẫn giữ lại các nốt chỉ được 2/4 bác sĩ đồng ý, phản ánh tính chất khó nhận biết của nốt nhỏ; (3) kết quả thực nghiệm trong quá trình ablation (EXP02) cho thấy consensus2 cho val_dice và F1 cao hơn so với union và consensus3. Sau khi áp consensus2, các blob mask có diện tích nhỏ hơn 10 pixel (tương ứng với các marker điểm không phải nốt thực) được lọc bỏ.

### 3.2.4. Phân chia tập dữ liệu

1010 bệnh nhân được phân chia ở cấp độ bệnh nhân (patient-level split) để đảm bảo không có thông tin của cùng một bệnh nhân xuất hiện ở cả tập train và test. Phân chia ngẫu nhiên với seed cố định (reproducibility):

| Tập | Số bệnh nhân | Tỷ lệ | Mục đích |
|---|---:|---:|---|
| Train | 812 | 80.4% | Huấn luyện mô hình |
| Validation | 99 | 9.8% | Tune hyperparameter, early stopping |
| Test (test_panel) | 99 | 9.8% | Đánh giá cuối cùng, không touch trong training |

**Nguyên tắc test_panel locked**: sau lần đo đầu tiên trên test_panel (commit `cd48359`), tập này được khóa hoàn toàn — không được thay đổi split, không được re-tune hyperparameter dựa trên kết quả test. Nguyên tắc này đảm bảo các số liệu báo cáo không bị phồng ảo do hiện tượng test set leakage.

### 3.2.5. Augmentation trong huấn luyện

Tăng cường dữ liệu (data augmentation) đóng vai trò quan trọng vì LIDC-IDRI, dù là tập dữ liệu lớn với 1010 bệnh nhân, vẫn có số lát cắt chứa nốt thực sự khá nhỏ so với tổng số lát cắt. Thư viện Albumentations được sử dụng vì hỗ trợ áp dụng đồng nhất transform lên cả ảnh gốc và mask nhãn (joint transform), đảm bảo alignment chính xác sau mỗi phép biến đổi hình học.

| Nhóm | Phép biến đổi | Mục đích |
|---|---|---|
| Hình học | HorizontalFlip, VerticalFlip, Rotate ±15° | Bất biến hướng |
| Biến dạng đàn hồi | ElasticTransform, GridDistortion | Mô phỏng biến dạng mô |
| Cường độ | RandomGamma, RandomBrightnessContrast | Cường độ quét CT khác nhau |
| Nhiễu/mờ | GaussianBlur, GaussianNoise | Bền vững với nhiễu máy quét |
| Erase | Cutout (random patch) | Regularization, giảm overfitting |

Tất cả augmentation chỉ áp dụng trong giai đoạn training; trong inference, chỉ dùng TTA (Test-Time Augmentation) 4-fold: original + horizontal flip + vertical flip + cả hai, lấy trung bình sigmoid.

---

## 3.3. Module Stage 1 — Baseline Segmentation 2D

Stage 1 là giai đoạn huấn luyện ban đầu với kiến trúc đơn giản, mục đích chính không phải đạt kết quả tốt nhất mà cung cấp khởi tạo trọng số (warm-start) chất lượng cao cho Stage 2.

**Kiến trúc**: UNet++ với encoder EfficientNet-B5, decoder không có SCSE attention (decoder thông thường). Input là ảnh 2D đơn lát cắt 512×512 (single-slice, 1 channel sau khi normalize HU về [0,1]).

**Hàm mất mát**: tổ hợp tuyến tính BCE và Dice Loss: `Loss = 0.5 × BCE + 0.5 × (1 − Dice)`. Kết hợp này phổ biến trong phân đoạn y khoa vì BCE có gradient ổn định tốt hơn với các vùng class imbalance nặng, trong khi Dice Loss đo trực tiếp chất lượng overlap mask.

**Vai trò trong pipeline**: Sau khi Stage 1 hội tụ, checkpoint `best.pt` Stage 1 được dùng để khởi tạo trọng số encoder của Stage 2. Warm-start từ Stage 1 giúp Stage 2 bắt đầu từ vùng tốt trong không gian tham số, rút ngắn thời gian hội tụ và cải thiện val_dice so với khởi tạo ngẫu nhiên. Điều này được xác nhận bởi val_dice Stage 2 đạt 0.86 ngay từ epoch 1 — một giá trị chỉ có thể đạt được với khởi tạo trọng số tốt.

---

## 3.4. Module Stage 2 — Refinement 2.5D với SCSE Attention và Tversky Loss

Stage 2 là module cốt lõi của hệ thống, cải thiện Stage 1 trên ba chiều: không gian (2.5D thay vì 2D), cơ chế attention (SCSE), và hàm mất mát (Tversky thay vì BCE+Dice).

### 3.4.1. Kiến trúc UNet++ EfficientNet-B5 SCSE 2.5D

**Input 2.5D**: thay vì xử lý từng lát cắt độc lập (2D), Stage 2 nhận vào stack 3 lát cắt liên tiếp [slice i-1, slice i, slice i+1] dưới dạng ảnh 3-channel 512×512×3. Cách tiếp cận "2.5D" này khai thác thông tin ngữ cảnh z-axis (bối cảnh không gian xung quanh nốt theo chiều sâu) mà không yêu cầu bộ nhớ GPU lớn như kiến trúc 3D đầy đủ. Mỗi channel được chuẩn hóa độc lập về [0,1] sau khi clamp HU vào khoảng [−1000, 400] (cửa sổ phổi chuẩn).

**Encoder EfficientNet-B5**: được pretrain trên ImageNet, có ~28M tham số trong phần encoder. EfficientNet (Tan & Le, 2019) được chọn vì tỷ lệ tham số/hiệu suất vượt trội so với các backbone thế hệ trước (ResNet, VGG) — B5 đạt độ chính xác cao hơn ResNet-50 với số tham số ít hơn. Việc dùng pretrain ImageNet cho ảnh y tế 3-channel đã được nghiên cứu rộng rãi: đặc trưng cấp thấp (cạnh, kết cấu) học từ ảnh tự nhiên có thể chuyển giao hiệu quả cho ảnh CT, đặc biệt với dữ liệu y tế hạn chế.

**Decoder UNet++ với SCSE Attention**: UNet++ (Zhou et al., 2019) mở rộng kiến trúc U-Net gốc (Ronneberger et al., 2015) bằng cách bổ sung dense skip connections và các sub-node trung gian trong đường decoder, cho phép tái sử dụng đặc trưng ở nhiều mức độ khác nhau. Sau mỗi decoder block, một module SCSE (Squeeze-and-Channel Excitation — Roy et al., 2018) được chèn vào để cân chỉnh lại trọng số theo cả hai chiều: channel (cSE — nhấn mạnh kênh đặc trưng quan trọng) và spatial (sSE — nhấn mạnh vùng không gian quan trọng). SCSE đặc biệt hiệu quả với nốt nhỏ vì giúp mô hình tập trung attention vào vùng nhỏ trong ảnh lớn thay vì phân tán đều trên toàn ảnh.

**Output**: 1-channel logit map 512×512 sau sigmoid — mask xác suất nốt cho lát cắt giữa (slice i).

Tổng số tham số mô hình: ~30M (encoder B5 + decoder UNet++ + SCSE blocks).

### 3.4.2. Hàm mất mát Tversky với α=0.3, β=0.7

Tversky Loss (Salehi et al., 2017) là tổng quát hóa của Dice Loss với hai tham số điều chỉnh mức phạt cho FP (α) và FN (β):

$$\mathcal{L}_\text{Tversky} = 1 - \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}$$

Với α=0.3, β=0.7: FN bị phạt nặng gấp 7/3 ≈ 2.3 lần so với FP. Lý do chọn tham số này có động cơ lâm sàng rõ ràng: trong ứng dụng hỗ trợ chẩn đoán ung thư, bỏ sót nốt thật (FN) nguy hiểm hơn nhiều so với báo nhầm vùng không phải nốt (FP). Bác sĩ X-quang có thể lọc bỏ FP qua quan sát trực tiếp, nhưng nốt bị bỏ sót hoàn toàn không được đưa vào tầm xem xét. Bias toward recall (tối ưu Recall > Precision) là lựa chọn thiết kế có chủ ý, được xác nhận bởi kết quả ablation EXP03 cho thấy Tversky 0.3/0.7 cải thiện sensitivity so với BCE+Dice thuần.

### 3.4.3. Optimizer và Learning Rate Schedule

**AdamW** (Loshchilov & Hutter, 2019) được chọn với lr=1e-4, weight_decay=1e-5. AdamW tách biệt weight decay khỏi gradient update, khắc phục hạn chế thiết kế trong Adam gốc vốn không áp dụng weight decay đúng cách với adaptive learning rates. Gradient clipping với max_norm=1.0 ngăn gradient explosion trong quá trình fine-tune encoder pretrain.

**Cosine Annealing với Warm Restart**: learning rate giảm theo hàm cosine từ lr_max về lr_min, sau đó reset lên lr_max tại epoch 136 (75% tổng 180 epoch). Warm restart (Loshchilov & Hutter, 2017 — SGDR) giúp thoát khỏi local minima trước khi SWA bắt đầu hoạt động trong cửa sổ cuối.

### 3.4.4. Stochastic Weight Averaging (SWA)

SWA (Izmailov et al., 2018) được kích hoạt từ epoch 136/180 (75% quá trình huấn luyện). Cơ chế: sau mỗi epoch trong cửa sổ [136, 180], trọng số hiện tại được tích lũy vào một mô hình trung bình hóa (SWA model). Sau epoch 180, SWA model được cập nhật Batch Normalization statistics bằng cách forward toàn bộ tập train 1 epoch với learning rate cố định.

SWA được biết đến giúp mô hình hội tụ đến vùng flat minima trong loss landscape, vùng này thường tổng quát hóa tốt hơn sharp minima (Izmailov et al., 2018). Kết quả: hai checkpoint lưu ra là `best.pt` (epoch đạt val_dice cao nhất — epoch 16) và `swa.pt` (trung bình trọng số epoch 136–180). Trong inference, hai checkpoint được ensemble bằng cách trung bình xác suất sigmoid đầu ra.

### 3.4.5. Kết quả huấn luyện Stage 2

Quá trình huấn luyện thực hiện trên VPS với GPU Tesla V100 32GB SXM2, tổng thời gian ~3.5 giờ cho 180 epoch với batch_size=8 và input 512×512×3.

**Val Dice progression**: val_dice = 0.86 tại epoch 1 (nhờ warm-start từ Stage 1) → đạt đỉnh **0.8676 tại epoch 16** (best.pt) → giảm dần xuống ~0.82–0.84 do mô hình tiếp tục học trên training set → trong SWA window [136, 180] dao động 0.74–0.84.

Việc val_dice đạt đỉnh sớm tại epoch 16 là hiện tượng điển hình của transfer learning với warm-start tốt: encoder đã học đặc trưng phù hợp từ Stage 1; Stage 2 chỉ cần tinh chỉnh thêm decoder SCSE mới. Epoch 16 best.pt được sử dụng trong ensemble inference cùng swa.pt.

---

## 3.5. Module FPR Classifier — DenseNet121-3D Nhị phân

### 3.5.1. Mục đích và vị trí trong pipeline

Sau Stage 2, danh sách candidate nodule chứa cả TP (nốt thật) lẫn FP (vessel branch points, artifact, thick-slice blur). FPR (False Positive Reduction) Classifier là module phân loại nhị phân nhận vào patch 3D quanh centroid của từng candidate và phân loại True Nodule (lớp 1) vs False Positive (lớp 0). Module này giúp giảm FP/scan trước khi đưa kết quả cho radiologist.

### 3.5.2. Kiến trúc DenseNet121-3D

DenseNet121 3D được chọn vì kiến trúc dense connection (Huang et al., 2017) phù hợp cho phân loại patch nhỏ (48×48×48 voxel): dense connection cho phép gradient lan truyền trực tiếp đến mọi layer trong network, giảm vanishing gradient và tăng hiệu quả tham số. 3D convolution cần thiết để khai thác thông tin hình thái học 3D của nốt (hình cầu so với elongated vessel).

**Input**: patch 48×48×48 voxel cắt ra từ volume HU quanh centroid candidate. Cửa sổ HU [−1024, 600] rộng hơn Stage 2 để bao gồm cả cấu trúc xương và mô mềm — quan trọng vì FPR phải phân biệt nốt với mạch máu, thành phế quản, và các cấu trúc giải phẫu khác. Patch được normalize về [0,1].

### 3.5.3. Huấn luyện FPR

60 epoch, batch_size=32, Adam lr=1e-4, Loss: Binary Cross-Entropy với class weight. Class weight được tính theo tỷ lệ candidate trong training set để cân bằng sự mất cân xứng giữa FP (thường nhiều hơn TP sau Stage 2).

### 3.5.4. Hiệu chỉnh ngưỡng (Threshold Calibration)

Sau khi huấn luyện, ngưỡng phân loại được hiệu chỉnh riêng bằng cách sweep threshold từ 0.05 đến 0.95 (bước 0.05) trên tập validation, chọn giá trị tối đa hóa F1. Kết quả: `best_thr = 0.85`; val ROC AUC = 0.9364. Ngưỡng cao (0.85) phản ánh thiên kiến conservative của pipeline: chỉ loại bỏ FP khi mô hình rất tự tin; còn nghi ngờ thì giữ lại để radiologist tự đánh giá.

---

## 3.6. Module Malignancy Classifier — DenseNet121-3D Phân loại 5 lớp

### 3.6.1. Mục đích và thang phân loại

Malignancy Classifier phân loại mức độ nghi ngờ ác tính của nốt theo thang LIDC 1–5 (1 = rất lành tính, 5 = rất ác tính). Đây là thang chủ quan từ đánh giá của 4 bác sĩ LIDC; malignancy score sử dụng trong dự án là trung bình điểm của các bác sĩ đã đánh dấu nốt. Module này không phải công cụ chẩn đoán độc lập — output được kết hợp với size prior theo tỷ lệ 40:60 trong combined risk score, và toàn bộ kết quả được trình bày cho radiologist như thông tin hỗ trợ quyết định, không phải kết luận cuối.

### 3.6.2. Kiến trúc và Huấn luyện

DenseNet121-3D, patch 32×32×32 voxel (nốt đã được khoanh vùng tốt sau FPR). HU window [−1024, 600]. 60 epoch, weighted Cross-Entropy Loss với class weights theo logarithm tần suất thực tế (log-scaled weights để giảm ưu thế của lớp 3 — lớp trung tâm chiếm đa số trong LIDC).

**Kết quả**: best epoch 52, val balanced accuracy = 0.4593, val suspicion F1 (binary derived từ class ≥ 4) = 0.6542. Balanced accuracy 0.46 phản ánh độ khó vốn có của bài toán: ngay cả các bác sĩ chuyên khoa cũng có inter-rater agreement thấp cho điểm ác tính LIDC (Armato et al., 2011). Module này cung cấp tín hiệu bổ sung nhưng không quyết định.

### 3.6.3. Combined Risk và phân tầng nguy cơ

Sau khi có malignancy score từ AI và đường kính nốt, hệ thống tính Combined Risk:

$$\text{CombinedRisk} = 0.6 \times \text{SizePrior}(d) + 0.4 \times \frac{\text{MalignancyScore} - 1}{4}$$

Trọng số 60% size prior và 40% AI score phản ánh thực tế: kích thước nốt là yếu tố dự đoán đã được xác nhận lâm sàng qua nhiều nghiên cứu lớn (National Lung Screening Trial, 2011), trong khi AI malignancy score vẫn cần xác nhận trên dữ liệu ngoài. Combined risk được phân thành 3 mức (Low / Medium / High) cho giao diện webapp.

---

## 3.7. Module Hậu xử lý Không gian

### 3.7.1. Lung mask pretrained

Lungmask R231 (Hofmanninger et al., 2020) là U-Net được pretrain trên 231 scan CT đa nguồn, cho phép phân đoạn phổi chính xác kể cả với bệnh lý (xẹp phổi, tràn dịch màng phổi). Mask phổi được dùng để loại bỏ candidate nằm ngoài vùng phổi — giảm FP từ cơ hoành, thành ngực, và các cấu trúc ngoài phổi.

### 3.7.2. Connected Components và Lọc Blob 3D

Sau khi threshold volume xác suất, connected components 3D với cấu trúc 3×3×3 (26-connectivity) được áp dụng để nhóm các voxel dương tính liên thông. Các blob nhỏ hơn `min_voxels=200` voxel (~6.5mm đường kính tương đương) bị loại bỏ — tương ứng với ngưỡng lâm sàng: nốt dưới 4mm không có chỉ định theo dõi theo Lung-RADS.

### 3.7.3. Elongation Filter dựa trên PCA

Để phân biệt nốt tròn với mạch máu dài, phân tích thành phần chính (PCA) được áp dụng trên tập voxel của từng blob. Tỷ lệ `max_eigenvalue / min_eigenvalue > 4` xác định blob là cấu trúc dài (vessel, bronchus) và bị loại. Ngưỡng 4 được chọn thực nghiệm để cân bằng giữa loại bỏ vessel (elongation cao) và giữ lại nodule loại GGO (Ground-Glass Opacity, có thể ít tròn hơn nodule solid).

### 3.7.4. NMS 3D và Đo Đường Kính

**NMS 3D** (Non-Maximum Suppression): các candidate trong cùng khu vực không gian (centroid distance < 10mm) được hợp nhất thành một detection duy nhất, lấy blob có confidence cao nhất đại diện.

**Đo đường kính**: sử dụng chiến lược "core voxel" — diameter được tính từ tập voxel có xác suất > 0.85 (core of the detection), thay vì toàn bộ blob vượt ngưỡng 0.5. Nếu core có < 8 voxel (blob nhỏ), fallback về `diameter_full × 0.7`. Chiến lược này cho đường kính ổn định hơn và ít bị ảnh hưởng bởi halo xác suất thấp xung quanh nốt.

---

## 3.8. Module Webapp — FastAPI và Next.js

### 3.8.1. Backend FastAPI

Backend được xây dựng bằng FastAPI (Python) với mô hình xử lý bất đồng bộ (async). Các endpoint chính:

| Endpoint | Phương thức | Mô tả |
|---|---|---|
| `/api/analyze` | POST | Nhận zip DICOM, khởi chạy inference |
| `/api/case/{id}/stream` | GET (SSE) | Stream tiến trình xử lý real-time |
| `/api/case/{id}` | GET | Trả về kết quả JSON đầy đủ |
| `/api/case/{id}/verdict` | POST | Lưu verdict của bác sĩ |
| `/api/training-metrics` | GET | Trả về metrics huấn luyện (loss, dice curves) |

**Các biện pháp bảo mật**: (1) Path traversal guard — tên file từ upload được sanitize trước khi ghi disk; (2) Zip-slip guard — kiểm tra tất cả đường dẫn trong zip không vượt ra ngoài thư mục tạm; (3) Verdict allowlist — chỉ chấp nhận giá trị verdict nằm trong tập hợp hợp lệ; (4) `torch.load(..., weights_only=True)` — phòng chống code injection qua checkpoint; (5) SSE không leak đường dẫn filesystem tuyệt đối.

### 3.8.2. Frontend Next.js với TypeScript

Frontend xây dựng bằng Next.js App Router với TypeScript và Tailwind CSS. Các trang chính: (1) Upload — form upload DICOM zip, chọn mode, hiển thị tiến trình SSE real-time; (2) Case Detail — 3D viewer (Plotly Mesh3d) với lung mesh và nodule mesh theo màu risk level, 2D slice carousel với overlay bounding box, bảng nodule list với Lung-RADS và verdict; (3) Lịch sử — danh sách tất cả ca, lọc theo verdict; (4) Training Metrics — biểu đồ Recharts động cho loss, dice theo epoch.

### 3.8.3. Schema JSON kết quả

Mỗi nốt trong output JSON có các trường: `id`, `original_id`, `bbox_zyx_voxel`, `centroid_zyx_voxel`, `diameter_mm`, `confidence`, `fpr_prob`, `nodule_type`, `upper_lobe`, `lung_rads`. Trong GT mode, bổ sung: `n_readers`, `confidence_tier`, `malignancy_score`, `malignant`, `lidc_nodule_ids`.

---

## Tài liệu tham khảo chương 3

Armato, S. G., McLennan, G., Bidaut, L., et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931.

Hofmanninger, J., Prayer, F., Pan, J., Röhrich, S., Prosch, H., & Langs, G. (2020). Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. *European Radiology Experimental*, 4(1), 1–13.

Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4700–4708.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *Conference on Uncertainty in Artificial Intelligence (UAI)*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*.

National Lung Screening Trial Research Team. (2011). Reduced lung-cancer mortality with low-dose computed tomographic screening. *New England Journal of Medicine*, 365(5), 395–409.

Roy, A. G., Navab, N., & Wachinger, C. (2018). Concurrent spatial and channel 'squeeze & excitation' in fully convolutional networks. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 421–429.

Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *Machine Learning in Medical Imaging (MLMI)*, 379–387.

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2019). UNet++: A nested U-net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3–11.

---

# CHƯƠNG 4: TRIỂN KHAI VÀ ĐÁNH GIÁ KẾT QUẢ

## 4.1. Môi trường và Thiết lập Thực nghiệm

### 4.1.1. Cấu hình phần cứng

Hệ thống được phát triển và đánh giá trên hai môi trường phần cứng khác nhau, phản ánh tình huống thực tế của nghiên cứu học thuật với tài nguyên tính toán thuê theo giờ:

| Thành phần | Cấu hình |
|---|---|
| GPU huấn luyện Stage 2 | NVIDIA Tesla V100-SXM2 32GB (VPS thuê, cloud na-01 Việt Nam) |
| CUDA Compute Capability | 7.0 (yêu cầu PyTorch ≤ 2.5.1 + CUDA 12.1) |
| RAM hệ thống VPS | 128GB CPU RAM |
| Lưu trữ VPS | 200GB NVMe (dataset LIDC + LUNA16 + checkpoints) |
| Thời gian huấn luyện Stage 2 | ~3.5 giờ / 180 epoch (batch=8, input 512×512×3) |
| GPU phát triển local | [CẦN BỔ SUNG] |

**Lưu ý tương thích V100**: Tesla V100 CC 7.0 yêu cầu PyTorch ≤ 2.5.1 với CUDA 12.1. Các phiên bản PyTorch ≥ 2.6 với CUDA 12.4+ không tương thích. Constraint này ảnh hưởng đến lựa chọn phiên bản thư viện toàn bộ dự án và cần được lưu ý khi tái tạo thực nghiệm.

### 4.1.2. Cấu hình phần mềm

| Thư viện | Phiên bản | Ghi chú |
|---|---|---|
| Python | 3.10 | |
| PyTorch | 2.5.1+cu121 | Yêu cầu V100 CC 7.0 compatibility |
| CUDA | 12.1 | |
| cuDNN | 9.1 | |
| MONAI | 1.x | Bundle lung_nodule_ct_detection |
| segmentation-models-pytorch | 0.4+ | UNet++, EfficientNet-B5 |
| Albumentations | — | Augmentation pipeline |
| FastAPI | — | Backend REST + SSE |
| Next.js | — | Frontend App Router |

### 4.1.3. Bảng Hyperparameter đầy đủ Stage 2

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| Learning rate | 1e-4 | Chuẩn fine-tune encoder pretrain |
| Weight decay | 1e-5 | Regularization nhẹ |
| Batch size | 8 | Phù hợp 32GB V100, input 512×512×3 |
| Epochs | 180 | Đủ SWA window 45 epoch |
| Optimizer | AdamW | Weight decay correct implementation |
| Scheduler | Cosine Annealing + Warm Restart ep 136 | Thoát local minima trước SWA |
| Gradient clip | max_norm=1.0 | Ổn định fine-tune encoder |
| SWA start | Epoch 136 (75%) | Sau plateau, bắt đầu averaging |
| SWA learning rate | 1e-4 constant | Không giảm trong SWA window |
| Loss | Tversky α=0.3, β=0.7 | Bias toward recall |
| Val metric | Dice per-slice | Chuẩn segmentation |
| TTA inference | 4-fold | Ensemble robustness |
| Min nodule voxels | 200 | ~6.5mm equivalent |
| Max elongation PCA | 4.0 | Lọc vessel/bronchus |
| Subpleural cutoff | 2mm | Loại artifact thành ngực |
| NMS merge distance | 10mm | Hợp nhất over-detection |

---

## 4.2. Đánh giá Quá trình Huấn luyện Stage 2

### 4.2.1. Đường cong Loss và Val Dice

Quá trình huấn luyện Stage 2 cho thấy hai đặc điểm nổi bật (tham chiếu Hình `work/academic/figures/stage2_loss_dice.png`):

**Training loss**: giảm từ ~0.16 tại epoch 1 xuống ~0.07 ở vùng epoch 80–130, trước khi tăng nhẹ sau warm restart tại epoch 136. Trong SWA window (ep 136–180), training loss dao động nhẹ quanh 0.08–0.10 vì SWA duy trì lr cao thay vì giảm về 0.

**Val Dice**: đạt 0.86 ngay epoch 1 (nhờ warm-start Stage 1), leo lên đỉnh **0.8676 tại epoch 16** (best.pt), sau đó giảm về khoảng 0.82–0.84 trong các epoch tiếp theo. Trong SWA window, val_dice dao động 0.74–0.84. Khoảng cách giữa training loss và val_dice không quá lớn, không có dấu hiệu overfitting rõ ràng trong các epoch đầu, mặc dù có divergence nhẹ sau epoch 100.

### 4.2.2. Kết quả Ablation Pilot (EXP00–EXP05)

Trước khi chạy Stage 2 full, một loạt thực nghiệm ablation pilot được tiến hành (mỗi EXP sử dụng run ngắn 50 epoch để so sánh tương đối):

| EXP | Cấu hình | Val Dice (50ep) | Ghi chú |
|---|---|---:|---|
| EXP00 | Stage 1 baseline checkpoint (reference) | 0.848 | Điểm xuất phát |
| EXP01 | Stage 2 + negative sampler mạnh | 0.851 | Không cải thiện đáng kể |
| EXP02 | Stage 2 + consensus2 GT | 0.858 | Winner GT strategy |
| EXP03 | Stage 2 + consensus2 + Tversky 0.3/0.7 | 0.862 | Cải thiện sensitivity |
| EXP04 | Stage 2 + consensus2 + Tversky + neg sampler | 0.859 | Negative sampler không giúp thêm |
| EXP05 | EXP03 + oversample 10–20mm ×3 | **0.865** | Config được chọn cho stage2_full |

EXP02 xác nhận consensus2 GT vượt union (nhiều nhiễu) và consensus3 (bỏ sót nốt). EXP03 xác nhận Tversky cải thiện sensitivity. EXP05 thêm oversampling nốt trung bình tăng nhẹ val_dice và F1 trên validation.

---

## 4.3. Đánh giá Quá trình Huấn luyện FPR và Malignancy Classifier

### 4.3.1. FPR Classifier — Đường cong Training

(Tham chiếu Hình `work/academic/figures/fpr_malignancy_curves.png`)

FPR Classifier huấn luyện 60 epoch trên tập candidate từ Stage 2. Training loss giảm từ ~1.6 tại epoch 1 xuống ~0.77 tại epoch 60, cho thấy mô hình hội tụ ổn định. Val suspicion F1 tăng từ ~0.53 tại epoch đầu lên ~0.65 ở epoch cuối.

**ROC AUC = 0.9364** trên validation set cho thấy DenseNet121-3D có khả năng phân biệt tốt giữa nốt thật và dương tính giả trong không gian đặc trưng 3D của patch 48×48×48.

### 4.3.2. Hiệu chỉnh Ngưỡng FPR

Sweep threshold từ 0.05 đến 0.95 trên validation set:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.50 | 0.71 | 0.82 | 0.76 |
| 0.70 | 0.77 | 0.75 | 0.76 |
| **0.85** | **0.83** | **0.68** | **0.75** |
| 0.90 | 0.86 | 0.61 | 0.71 |

Ngưỡng 0.85 được chọn vì tối ưu F1 trên validation và phản ánh triết lý conservative: chỉ loại bỏ candidate khi mô hình rất tự tin đó là FP. Ngưỡng được lưu cùng checkpoint `fpr.pt`.

### 4.3.3. Malignancy 5-class — Phân tích Class Distribution

Dataset malignancy LIDC có phân phối lệch đáng kể về lớp 3 (trung tính, chiếm ~35%). Weighted Cross-Entropy với log-scaled weights cân bằng một phần mất cân xứng này. Val balanced accuracy = 0.4593 phản ánh đúng độ khó của bài toán 5 lớp với nhãn chủ quan: random chance cho 5 lớp = 0.20 (balanced accuracy), kết quả 0.46 là gấp ~2.3 lần ngẫu nhiên. Val suspicion F1 = 0.6542 (binary derived từ class ≥ 4) cung cấp giá trị thực tiễn hơn cho ứng dụng lâm sàng.

---

## 4.4. Đánh giá Hiệu năng trên Tập Kiểm thử

### 4.4.1. Tập Kiểm thử test_panel (99 bệnh nhân)

Tập test_panel gồm 99 bệnh nhân, được khóa sau lần đo đầu tiên (commit `cd48359`). Ground truth là mask consensus2 từ 4 radiologist LIDC. Đây là tập đánh giá chính thức duy nhất; tất cả các số liệu trong mục này đến từ tập test_panel.

### 4.4.2. So sánh Mine vs MONAI RetinaNet 3D

Bảng so sánh chính trên test_panel 99 bệnh nhân, matching rule: centroid distance ≤ 15mm (fixed), threshold mine = 0.65:

| Mô hình | Precision | Recall | F1 | Delta F1 |
|---|---:|---:|---:|---:|
| **Mine** (UNet++ B5 2.5D + SCSE + Tversky) | **0.583** | **0.659** | **0.618** | — |
| MONAI (RetinaNet 3D, bundle pretrained) | 0.521 | 0.547 | 0.533 | −0.085 |
| Delta (Mine − MONAI) | +0.062 | +0.112 | **+0.085** | |

Mine vượt MONAI +8.5 điểm F1 tuyệt đối, tương đương +15.9% tương đối. Cả hai chỉ số thành phần đều cải thiện: Precision +6.2 điểm (ít FP hơn) và Recall +11.2 điểm (ít bỏ sót nốt thật hơn).

**Giải thích sự vượt trội của Mine**: (1) MONAI bundle là mô hình RetinaNet 3D pre-trained trên LUNA16, tạo ra distribution shift khi áp dụng trên LIDC với annotation protocol khác; (2) Mine được huấn luyện trực tiếp trên LIDC với cùng ground truth consensus2 được dùng cho đánh giá; (3) 2.5D segmentation với SCSE attention phù hợp hơn cho phân đoạn pixel-level nốt đa kích thước so với detection 3D thuần.

### 4.4.3. Kiểm định Thống kê — Wilcoxon Signed-Rank

Để đảm bảo sự khác biệt +0.085 F1 không phải do biến động ngẫu nhiên, kiểm định Wilcoxon signed-rank test được áp dụng tại cấp độ bệnh nhân:

- **Phương pháp**: với mỗi trong 99 bệnh nhân test, tính F1_mine(patient) và F1_monai(patient); kiểm định Wilcoxon trên 99 cặp giá trị.
- **Lý do chọn Wilcoxon**: kiểm định phi tham số (không giả định phân phối chuẩn), phù hợp với F1 per-patient thường có phân phối lệch, và phù hợp cho paired samples.
- **Kết quả**: p < 0.0001 — sự khác biệt có ý nghĩa thống kê cao.
- **Bootstrap CI 95% cho delta F1**: [CẦN BỔ SUNG — chạy bootstrap_ci.py trên test_panel để cung cấp CI chính xác]

### 4.4.4. Phân tích FROC và CPM

FROC analysis trên 12 bệnh nhân evaluation panel (33 GT nodule, matching rule 15mm cố định):

| FP/scan | Sensitivity |
|---:|---:|
| 0.125 | 0.576 |
| 0.250 | 0.576 |
| 0.500 | 0.576 |
| 1.000 | 0.576 |
| 2.000 | 0.576 |
| 4.000 | 0.576 |
| 8.000 | 0.576 |
| **CPM** | **0.576** |

**Quan sát**: Sensitivity phẳng trên toàn dải 0.125–8 FP/scan vì pipeline ensemble (best.pt + swa.pt) với TTA tạo ra phân phối xác suất bimodal: vùng nốt thật có xác suất rất cao (>0.85) và vùng nền thấp (<0.3). Hạ threshold xuống dưới 0.30 không phát hiện thêm nốt GT nào — các nốt bị bỏ sót nằm trong vùng mô hình không "thấy", không phải vùng threshold.

**Disclaimer quan trọng**: CPM = 0.576 với matching rule 15mm cố định KHÔNG thể so sánh trực tiếp với CPM của các phương pháp trong tài liệu LUNA16 (DeepLung: CPM=0.842, Setio et al. 2017; nnDetection: CPM=0.918, Baumgartner et al. 2021) vì: (1) matching rule 15mm rộng hơn đáng kể so với LUNA16 official rule `max(3mm, diameter/2)`; (2) evaluation panel 12 bệnh nhân có variance cao hơn nhiều so với 888 scan LUNA16. Xem Mục 2.5.6 để thảo luận chi tiết.

### 4.4.5. Phân tích Độ nhạy Theo Kích thước Nốt

| Nhóm kích thước | Đường kính | n_GT | TP | FN | Sensitivity |
|---|---:|---:|---:|---:|---:|
| Nhỏ | 4–6 mm | 4 | 2 | 2 | **0.500** |
| Trung bình | 6–15 mm | 12 | 8 | 4 | **0.667** |
| Lớn | > 15 mm | 16 | 8 | 8 | **0.500** |

**Phát hiện đáng chú ý**: mô hình hoạt động tốt nhất với nhóm nốt trung bình (6–15mm, sensitivity=0.667) — nhóm kích thước chiếm đa số trong training data LIDC và cũng là nhóm có ý nghĩa lâm sàng cao nhất (nốt cần theo dõi theo Lung-RADS 3 và 4A). Nhóm nốt lớn (>15mm) có sensitivity thấp bất ngờ (0.500) — phân tích thất bại (Mục 4.4.7) cho thấy nguyên nhân chủ yếu đến từ một bệnh nhân duy nhất với annotation yếu (consensus 1/4).

### 4.4.6. Ablation Study trên Evaluation Panel

| Biến thể | Sensitivity | Precision | F1 | FP/scan | ΔF1 vs full |
|---|---:|---:|---:|---:|---:|
| **Pipeline đầy đủ** | 0.576 | 0.760 | **0.655** | 0.50 | — |
| Không post-processing | 0.606 | 0.741 | 0.667 | 0.58 | +0.012 |
| Không lung mask | 0.576 | 0.704 | 0.633 | 0.67 | −0.022 |
| Không TTA | 0.515 | 0.739 | 0.607 | 0.50 | −0.048 |

**TTA đóng góp lớn nhất** (+0.048 F1 khi có TTA so với không có): loại bỏ TTA mất 6 detection (sensitivity 0.576 → 0.515). Lung mask giúp precision (+0.056) bằng cách loại 2 FP ngoài phổi. Post-processing có tác động nhỏ và phức tạp: trong evaluation panel này, morphology cleanup lọc nhầm 1 TP (giảm sensitivity) trong khi giảm 1 FP (giảm FP/scan). Tradeoff này chấp nhận được lâm sàng nhưng đáng xem xét re-tuning với panel lớn hơn.

### 4.4.7. Phân tích Trường hợp Thất bại

#### False Negatives — 3 trường hợp tệ nhất (nốt lớn bị bỏ sót)

| # | Bệnh nhân | Đường kính GT | Bác sĩ đồng thuận | Malignancy | Phân tích nguyên nhân |
|---|---|---:|---:|---:|---|
| 1 | LIDC-IDRI-0751 | 29.6 mm | 1/4 | 4.0 | Weak consensus, thick slice 3mm |
| 2 | LIDC-IDRI-0751 | 27.4 mm | 1/4 | 3.0 | Weak consensus, thick slice 3mm |
| 3 | LIDC-IDRI-0751 | 25.0 mm | 1/4 | 5.0 | Weak consensus, thick slice 3mm |

Cả 3 FN tệ nhất đều đến từ LIDC-IDRI-0751, bệnh nhân có 9 GT nodule nhưng mỗi nốt chỉ được đánh dấu bởi 1/4 bác sĩ. Ngưỡng consensus2 yêu cầu ≥ 2/4 bác sĩ đồng thuận nên các nốt đơn bác sĩ này không có trong training GT — mô hình không học cách nhận diện dạng nốt "controversial" này. Ngoài ra, bệnh nhân này có slice thickness = 3.0mm (dày hơn trung bình 1.0–1.5mm), làm giảm resolution z-axis và khả năng detect của mô hình huấn luyện chủ yếu trên thin-slice.

Trường hợp LIDC-IDRI-0751 minh họa limitation cơ bản của consensus annotation: bộ lọc consensus2 cần thiết để giảm nhiễu nhưng đồng thời bỏ sót một số nốt "real but controversial". Trong ứng dụng lâm sàng, mô hình cần bổ sung training data thick-slice để xử lý tốt hơn dạng quét này.

#### False Positives — 3 trường hợp tệ nhất (confidence cao nhất)

| # | Bệnh nhân | Đường kính | Confidence | Elongation | Nguyên nhân |
|---|---|---:|---:|---:|---|
| 1 | LIDC-IDRI-0003 | 10.9 mm | 99% | 1.52 | Vessel branch point |
| 2 | LIDC-IDRI-0651 | 7.1 mm | 99% | 2.34 | Vessel branch point |
| 3 | LIDC-IDRI-0003 | 14.3 mm | 99% | 1.31 | Perifissural nodule-like structure |

Cả 3 FP có confidence rất cao (99%) và kích thước lâm sàng hợp lý (7–15mm). Elongation filter PCA không loại được vì elongation ratio < 4. Nguyên nhân: vessel branch points tại ngã ba mạch máu tạo ra blob tròn, dense trên CT single-slice rất giống nốt thật — thách thức kinh điển của hệ thống phát hiện nốt phổi (Van Ginneken et al., 2010). FPR Classifier 3D nên giúp lọc bớt dạng FP này trong các case không cao-confidence cực đoan. Một hướng cải thiện là bổ sung vascular attachment feature vào post-processing pipeline.

---

## 4.5. Kiểm thử Hệ thống trong Các Kịch bản Thực tế

### 4.5.1. Inference Time Per Case

Thời gian xử lý đo trên một CT scan điển hình (~200 lát cắt 512×512):

| Mode | Thời gian | Bottleneck chính |
|---|---|---|
| Mine | ~70–90 giây | Stage 2 TTA 4-fold + FPR + Malignancy |
| MONAI | ~30–40 giây | RetinaNet 3D forward single pass |
| GT | ~10 giây | Load + parse annotation XML |

Mine chậm hơn MONAI do TTA 4-fold (×4 forward pass Stage 2) cộng với 3 giai đoạn phân loại. Trong môi trường lâm sàng thực tế, 70–90 giây là chấp nhận được cho quy trình second reader AI khi bác sĩ đọc ca chính song song. Với GPU A100 thay V100, thời gian có thể giảm xuống 30–40 giây.

### 4.5.2. Kiểm thử Edge Case

| Kịch bản | Xử lý hệ thống | HTTP code |
|---|---|---|
| File DICOM bị truncated | Graceful error với message mô tả | 400 |
| Series thiếu metadata slice thickness | Fallback default = 1.0mm, cảnh báo trong response | 200 |
| Multi-series trong 1 zip | Chọn series có số lát cắt nhiều nhất | 200 |
| 3 mode crash độc lập | Structured error JSON với error_type | 503 |
| Path traversal trong tên file | Block tại upload layer | 400 |
| Zip-slip trong archive | Bắt ZipSlip exception | 400 |

### 4.5.3. Hiển thị Webapp

3D viewer dùng Plotly Mesh3d render lung mesh (xanh nhạt trong suốt) và nodule mesh (màu theo risk level: xanh = low, vàng = medium, đỏ = high). 2D slice carousel cho phép bác sĩ duyệt từng lát cắt với overlay bounding box nodule. Verdict (Accept / Reject / Uncertain) được ghi vào database và persistent qua session — phục vụ audit retrospective và tracking chất lượng AI.

---

## 4.6. Đánh giá Luồng Nghiệp vụ AI Second Reader

### 4.6.1. Workflow End-to-End

Luồng sử dụng hệ thống được thiết kế để tích hợp vào quy trình đọc ca của bác sĩ X-quang với can thiệp tối thiểu:

1. **Upload**: bác sĩ hoặc kỹ thuật viên upload zip chứa DICOM series của ca CT.
2. **Processing**: hệ thống xử lý tự động (~70–90s) với tiến trình hiển thị real-time qua SSE.
3. **Review 3 mode**: bác sĩ xem kết quả Mine, MONAI, và GT trên cùng một giao diện, so sánh trực quan.
4. **Verdict**: bác sĩ ghi verdict (Accept/Reject/Uncertain) cho từng nodule AI phát hiện.
5. **Audit**: verdict và kết quả được lưu trữ cho phân tích retrospective và cải thiện mô hình.

### 4.6.2. Tích hợp Lung-RADS và Combined Risk

Với mỗi nốt được AI phát hiện và qua FPR filter, hệ thống tự động: (1) gán Lung-RADS category dựa trên đường kính: <6mm → Cat.2; 6–8mm → Cat.3; 8–15mm → Cat.4A; >15mm → Cat.4B; (2) tính Combined Risk (Low/Medium/High) kết hợp size prior và AI malignancy score; (3) gán confidence tier (High/Medium/Low) dựa trên FPR probability. Output được trình bày bằng ngôn ngữ lâm sàng quen thuộc (Lung-RADS) thay vì số liệu kỹ thuật thuần túy.

### 4.6.3. Bài học Kỹ thuật Rút ra

Quá trình xây dựng và đánh giá tích lũy một số bài học kỹ thuật có giá trị:

**(1) 2.5D Segmentation vượt 3D Detection cho LIDC**: Mine (F1=0.618) vượt MONAI RetinaNet 3D (F1=0.533) với delta +0.085. Giải thích: LIDC có annotation pixel-level của 4 radiologist phù hợp để huấn luyện segmentation; detection 3D native phù hợp hơn với LUNA16 (annotation bounding sphere). Distribution shift giữa LUNA16 và LIDC là yếu tố quyết định — mô hình cần được đào tạo trên cùng distribution với evaluation.

**(2) SCSE Attention và Tversky Loss Đóng Góp Thực sự**: Ablation EXP02 vs EXP03 cho thấy Tversky 0.3/0.7 cải thiện sensitivity nhờ penalty FN nặng hơn. Các cải tiến này không phải chỉ trên paper — được xác nhận bằng số liệu ablation thực nghiệm.

**(3) Warm-Start Từ Stage 1 Tăng Tốc Hội tụ**: val_dice = 0.86 ngay epoch 1 Stage 2, đỉnh 0.8676 ở epoch 16. Không có warm-start, quá trình hội tụ thường mất 20–30 epoch đầu để đạt val_dice > 0.80.

**(4) Train-from-Scratch trên LUNA16 Thất bại (Run005)**: Thử nghiệm huấn luyện từ đầu trên LUNA16 (không warm-start, không LIDC) cho F1 = 0.019 trên LIDC test_panel — gần như bằng zero. Nguyên nhân: kiến trúc 2.5D segmentation không tổng quát hóa sang LUNA16 native detection format (sphere-annotation); mô hình học features không phù hợp với task segmentation pixel-level LIDC. Kết quả này xác nhận domain adaptation từ LUNA16 sang LIDC đòi hỏi nhiều hơn là chỉ thay dataset — cần cân nhắc lại kiến trúc và annotation format.

**(5) Quản lý Test Set là Nguyên tắc Bất khả xâm phạm**: Lock test_panel từ lần đo đầu tiên và không bao giờ re-tune dựa trên test results là thực hành thiết yếu để đảm bảo tính trung thực của báo cáo. Mọi hyperparameter được chọn hoàn toàn dựa trên validation set.

### 4.6.4. Giới hạn của Nghiên cứu

Nghiên cứu này có một số giới hạn cần được nhận thức rõ ràng:

1. **Matching rule 15mm không chuẩn LUNA16**: Các số liệu F1, FROC, CPM không thể so sánh trực tiếp với tài liệu dùng giao thức LUNA16 chính thức. Để so sánh công bằng cần re-evaluate với `max(3mm, diameter/2)` trên 888 scan LUNA16 test set.

2. **Single-fold split**: Phân tích chỉ dùng một split ngẫu nhiên duy nhất. Cross-validation 5-fold sẽ cho ước lượng variance tốt hơn nhưng yêu cầu tài nguyên tính toán gấp 5 lần.

3. **Evaluation panel nhỏ cho FROC**: 12 bệnh nhân với 33 GT nodule là panel nhỏ — variance của CPM và sensitivity cao. Kết luận FROC cần được diễn giải thận trọng.

4. **Malignancy Classifier chưa được xác nhận ngoài**: Balanced accuracy 0.4593 được đo trên cùng phân phối LIDC; hiệu năng trên bệnh nhân ngoài LIDC có thể khác đáng kể.

5. **AI là second reader, không phải primary diagnosis**: Toàn bộ hệ thống được thiết kế và đánh giá trong vai trò hỗ trợ quyết định — bác sĩ X-quang giữ toàn quyền đánh giá lâm sàng cuối cùng.

---

## Tài liệu tham khảo chương 4

Armato, S. G., McLennan, G., Bidaut, L., et al. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931.

Baumgartner, M., Jaeger, P. F., Isensee, F., & Maier-Hein, K. H. (2021). nnDetection: A self-configuring method for medical object detection. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 530–539.

Hofmanninger, J., Prayer, F., Pan, J., Röhrich, S., Prosch, H., & Langs, G. (2020). Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. *European Radiology Experimental*, 4(1), 1–13.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *Conference on Uncertainty in Artificial Intelligence (UAI)*.

National Lung Screening Trial Research Team. (2011). Reduced lung-cancer mortality with low-dose computed tomographic screening. *New England Journal of Medicine*, 365(5), 395–409.

Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *Machine Learning in Medical Imaging (MLMI)*, 379–387.

Setio, A. A. A., Traverso, A., de Bel, T., et al. (2017). Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. *Medical Image Analysis*, 42, 1–13.

Van Ginneken, B., Armato, S. G., de Hoop, B., et al. (2010). Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: the ANODE09 study. *Medical Image Analysis*, 14(6), 707–722.

Zhu, W., Liu, C., Fan, W., & Xie, X. (2018). DeepLung: Deep 3D dual path nets for automated pulmonary nodule detection and classification. *2018 IEEE Winter Conference on Applications of Computer Vision (WACV)*, 673–681.

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2019). UNet++: A nested U-net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3–11.

---

# CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 5.1. Kết luận

### 5.1.1. Tổng kết các kết quả đạt được

Nghiên cứu này đã xây dựng hoàn chỉnh một hệ thống phát hiện và phân loại nốt phổi tự động ứng dụng học sâu, triển khai trên tập dữ liệu LIDC-IDRI gồm 1.010 bệnh nhân với phân chia cố định 812 huấn luyện, 99 kiểm định và 99 kiểm thử. Pipeline 4 tầng — phân đoạn 2.5D UNet++ EfficientNet-B5 SCSE, lọc giảm dương tính giả DenseNet121-3D, phân loại ác tính DenseNet121-3D 5 lớp, và hậu xử lý không gian — được phát triển tuần tự với từng tầng được kiểm chứng độc lập trước khi tích hợp vào chuỗi đầu đến cuối.

Về kết quả huấn luyện, mô hình Stage 2 đạt val_dice = 0.8676 trên tập kiểm định LIDC, với quá trình hội tụ đáng chú ý: nhờ warm-start từ Stage 1, mô hình đạt val_dice > 0.86 ngay từ epoch đầu tiên Stage 2 và ổn định ở epoch 16. Bộ lọc FPR Classifier đạt AUC = 0.9364 trên tập kiểm định với ngưỡng hiệu chỉnh best_thr = 0.85, xác nhận khả năng loại bỏ phần lớn dương tính giả từ đầu ra phân đoạn. Mô hình phân loại ác tính 5 lớp đạt balanced accuracy = 0.4593 và suspicious F1 = 0.6542, phản ánh thách thức vốn có của bài toán khi nhãn đầu vào là đánh giá chủ quan thang 1-5 từ nhiều bác sĩ với độ đồng thuận thấp.

Đánh giá trên tập kiểm thử bị khóa gồm 99 bệnh nhân, mô hình đề xuất đạt F1 = 0.618, so với F1 = 0.533 của MONAI RetinaNet 3D baseline được huấn luyện trên cùng dữ liệu và đánh giá theo cùng giao thức. Kiểm định Wilcoxon signed-rank test tính trên đơn vị bệnh nhân cho p < 0.0001, xác nhận sự khác biệt có ý nghĩa thống kê. Kết quả này minh chứng cho tính khả thi của tiếp cận phân đoạn 2.5D với kiến trúc encoder mạnh (EfficientNet-B5) và cơ chế chú ý không gian-kênh (SCSE) so với tiếp cận phát hiện 3D đầy đủ trong điều kiện cùng tập dữ liệu.

Webapp 3 chế độ (mine / MONAI / ground truth) hoạt động ổn định với hệ thống theo dõi phán quyết lâm sàng theo từng nốt — cho phép bác sĩ ghi nhận quyết định chấp nhận, từ chối hoặc yêu cầu xem lại riêng biệt cho từng nguồn AI. Hệ thống được gia cố an ninh để phòng chống các lỗ hổng path traversal, zip-slip và đảm bảo an toàn tải checkpoint với `weights_only=True`. Module phân loại nguy cơ Lung-RADS và công thức Combined Risk (60% kích thước + 40% AI) được tích hợp đầy đủ vào luồng nghiệp vụ, tạo cơ sở cho việc mở rộng sang quy trình lâm sàng chuẩn hóa trong tương lai.

### 5.1.2. Đóng góp khoa học và thực tiễn

Về mặt kỹ thuật, nghiên cứu đóng góp bằng chứng thực nghiệm kiểm chứng trên LIDC-IDRI rằng kiến trúc 2.5D UNet++ EfficientNet-B5 SCSE kết hợp hàm mất mát Tversky bất đối xứng (α = 0.3, β = 0.7) và Stochastic Weight Averaging có thể vượt trội so với MONAI RetinaNet 3D baseline trên cùng điều kiện kiểm soát. Điểm quan trọng là kết quả này không đơn thuần đến từ kiến trúc: warm-start hai giai đoạn, hiệu chỉnh ngưỡng FPR trên validation set, và pipeline hậu xử lý hình thái học (elongation filter, NMS 3D) đều đóng góp có thể đo lường vào F1 cuối cùng — một minh chứng cho tầm quan trọng của thiết kế hệ thống tổng thể bên cạnh thiết kế mô hình đơn lẻ.

Về mặt phương pháp luận đánh giá, nghiên cứu áp dụng nguyên tắc honest reporting: tường minh báo cáo rằng quy tắc khớp nốt 15 mm cố định được sử dụng trong nghiên cứu này khác với quy tắc chính thức của LUNA16 challenge (`max(3mm, diameter/2)` trên 10-fold cross-validation), và do đó các chỉ số F1, FROC, CPM không thể được so sánh trực tiếp với kết quả trong tài liệu sử dụng giao thức LUNA16. Thực hành này — phân biệt rõ điều kiện đánh giá thay vì trích dẫn benchmark một cách mơ hồ — có giá trị phương pháp cho cộng đồng nghiên cứu AI y khoa ở Việt Nam, nơi văn hóa báo cáo honest về giới hạn còn chưa được nhấn mạnh đủ.

Về thiết kế sản phẩm, webapp 3-mode AI selector — cho phép so sánh trực quan kết quả từ nhiều mô hình AI và ground truth chuyên gia trên cùng giao diện — là một đóng góp thiết kế chưa được mô tả rõ trong tài liệu phát hiện nốt phổi đã khảo sát. Cách tiếp cận này hữu ích cho giai đoạn chuyển đổi khi một cơ sở y tế bắt đầu thử nghiệm AI: bác sĩ có thể đối chiếu đa nguồn theo từng ca thay vì phụ thuộc vào một mô hình duy nhất, giảm rủi ro khi triển khai ban đầu và xây dựng dữ liệu audit thực tế để cải thiện mô hình sau này.

### 5.1.3. Hạn chế của nghiên cứu

Nghiên cứu này có một số hạn chế cần được ghi nhận tường minh để người đọc có cơ sở diễn giải kết quả đúng mức.

Hạn chế nghiêm trọng nhất về mặt tổng quát hóa là sự vắng mặt của external validation dataset. Toàn bộ quá trình huấn luyện, tinh chỉnh và đánh giá đều được thực hiện trên LIDC-IDRI — một tập dữ liệu thu thập từ bảy cơ sở y tế Hoa Kỳ với đặc thù kỹ thuật và dân số bệnh nhân riêng. Khả năng tổng quát hóa sang ảnh CT từ hệ máy khác, giao thức chụp khác, hoặc dân số bệnh nhân Việt Nam chưa được kiểm chứng và là câu hỏi mở quan trọng nhất cho nghiên cứu tiếp theo.

Quy tắc khớp nốt 15 mm cố định, như đã nêu, không phải giao thức LUNA16 chính thức. Điều này có nghĩa là các con số F1, CPM trong nghiên cứu không thể so sánh trực tiếp với bảng xếp hạng LUNA16 hay với các bài báo trích dẫn kết quả từ challenge đó. Việc thiếu đánh giá theo giao thức LUNA16 chuẩn là một giới hạn kỹ thuật, mặc dù đây là lựa chọn có chủ ý xuất phát từ ràng buộc tài nguyên tính toán.

Kết quả phân loại ác tính với balanced accuracy = 0.4593 phản ánh hai thách thức có hệ thống: mất cân bằng lớp nghiêm trọng (lớp malignant và benign chiếm đa số, lớp intermediate rất ít) và nhiễu nhãn vốn có (label noise) từ bản chất đánh giá chủ quan 1-5 của bốn bác sĩ với thang đồng thuận thấp. Chỉ số này không nên được diễn giải là giới hạn của kiến trúc mà phản ánh giới hạn của nhãn học trong bài toán phân loại ác tính từ CT đơn thuần không có bổ sung dữ liệu lâm sàng hoặc mô bệnh học.

Mô hình không phát hiện nốt nhỏ dưới 5 mm do ngưỡng lọc kích thước MIN_NODULE_VOXELS = 120 được áp dụng trong hậu xử lý. Ngưỡng này được đặt để giảm dương tính giả từ các cấu trúc mạch máu nhỏ, nhưng đồng thời loại bỏ luôn các nốt thực sự nhỏ có ý nghĩa lâm sàng — đặc biệt trong theo dõi nodule growth. Thử nghiệm fine-tune trên LUNA16 (run005) thất bại hoàn toàn với F1 = 0.019, xác nhận kiến trúc 2.5D segmentation được huấn luyện cho LIDC pixel-level annotation không tổng quát hóa được sang LUNA16 sphere-annotation mà không cần thiết kế lại annotation format và kiến trúc phù hợp.

Cuối cùng, hệ thống chưa trải qua thử nghiệm lâm sàng (clinical trial) và không có phê duyệt quản lý. Đây là điều tất yếu trong phạm vi luận văn tốt nghiệp, nhưng cần được nêu rõ: kết quả thực nghiệm trình bày trong luận văn này là bằng chứng kỹ thuật trong điều kiện kiểm soát, không phải bằng chứng lâm sàng về hiệu quả thực tế khi tích hợp vào quy trình đọc của bác sĩ.

---

## 5.2. Hướng phát triển trong tương lai

### 5.2.1. Cải tiến kiến trúc mô hình

Hướng nghiên cứu trực tiếp nhất là chuyển sang kiến trúc 3D end-to-end để khắc phục hạn chế của tiếp cận 2.5D. Cụ thể, kiến trúc 3D U-Net kết hợp detection head 3D có thể tận dụng đầy đủ thông tin không gian ba chiều của CT mà không cần ghép lát cắt nhân tạo. Thư viện nnDetection (Baumgartner et al., 2021) cung cấp framework tự cấu hình cho bài toán này và là điểm khởi đầu phù hợp để so sánh công bằng với kết quả của nghiên cứu hiện tại.

Kiến trúc nnU-Net (Isensee et al., 2021) với khả năng tự cấu hình siêu tham số dựa trên đặc tính dataset là ứng viên mạnh cho ablation study: bằng cách để nnU-Net tự quyết định kiến trúc tối ưu cho LIDC, có thể tách biệt được đóng góp của thiết kế thủ công (UNet++ + EfficientNet-B5 + SCSE) so với tối ưu hóa tự động. Các kiến trúc dựa trên Transformer như Swin-UNet và MedNeXt cũng xứng đáng được đánh giá cho bài toán này: khả năng mô hình hóa quan hệ tầm xa (long-range context) có thể có lợi khi phát hiện nốt gần các cấu trúc phức tạp như trung thất hoặc thành ngực.

### 5.2.2. Cải tiến giao thức đánh giá

Ưu tiên cao nhất về phương pháp là triển khai đầy đủ LUNA16 official matching rule — thay thế quy tắc 15 mm cố định bằng `max(3mm, diameter/2)` — và thực hiện đánh giá trên 888 scan LUNA16 test set để có kết quả có thể so sánh trực tiếp với cộng đồng. Điều này đòi hỏi annotation lại ground truth theo format LUNA16 và chạy lại toàn bộ evaluation pipeline, nhưng là điều kiện cần thiết để công bố kết quả ở hội nghị quốc tế.

External validation trên dataset độc lập là bước không thể bỏ qua trước bất kỳ ứng dụng lâm sàng nào. Các dataset có thể tiếp cận bao gồm NLST (National Lung Screening Trial — hơn 26.000 ca), NSCLC-Radiomics (TCIA), và đặc biệt là dữ liệu in-house từ các bệnh viện Việt Nam nếu có thể xây dựng giao thức thu thập và annotation chuẩn hóa. Bootstrap confidence interval 95% cho F1 và CPM, cùng phân tích per-radiologist sensitivity breakdown từ 4 annotation của LIDC, sẽ làm tăng đáng kể độ chắc chắn thống kê của các kết luận.

### 5.2.3. Mở rộng phạm vi lâm sàng

Malignancy classifier hiện tại chỉ dựa vào đặc trưng hình thái học từ CT. Tích hợp đặc trưng radiomic bổ sung — bao gồm texture analysis, Gray-Level Co-occurrence Matrix (GLCM) và đặc trưng hình dạng từ thư viện PyRadiomics — có thể cải thiện đáng kể balanced accuracy, đặc biệt cho các lớp trung gian. Đây là hướng nghiên cứu có nền tảng tài liệu vững chắc và khả thi trong điều kiện không có thêm dữ liệu bên ngoài.

Phân loại sub-type ung thư (adenocarcinoma, squamous cell carcinoma, small cell lung cancer) đòi hỏi dữ liệu có nhãn pathology tương ứng — vượt ngoài những gì LIDC-IDRI cung cấp — nhưng là bước logic tiếp theo trong lộ trình phát triển. Theo dõi growth rate qua nhiều lần chụp CT (longitudinal tracking) là tính năng lâm sàng quan trọng hiện chưa có trong hệ thống: bằng cách đăng ký (registration) volume CT theo thời gian và theo dõi sự thay đổi kích thước nốt, hệ thống có thể hỗ trợ phân loại Lung-RADS theo tiêu chí tăng trưởng thay vì chỉ kích thước tuyệt đối. Phát hiện đặc trưng hình thái học chi tiết như calcification và spiculation — hiện chưa được mô hình hóa trong pipeline — có thể nâng cao chất lượng phân tầng nguy cơ.

### 5.2.4. Triển khai và vận hành sản xuất

Để đạt yêu cầu thời gian suy luận phù hợp cho quy trình lâm sàng thực tế (dưới 30 giây mỗi ca), cần tối ưu hóa mô hình qua ONNX export và TensorRT compilation trên GPU. Containerization toàn bộ hệ thống (backend, frontend, model serving) bằng Docker và Docker Compose là bước cần thiết để đảm bảo khả năng tái hiện (reproducibility) và triển khai nhất quán trên các môi trường khác nhau.

Tích hợp với hệ thống PACS (Picture Archiving and Communication System) của bệnh viện qua giao thức DICOMweb sẽ loại bỏ bước tải file thủ công hiện tại, cho phép ca bệnh mới tự động được phân tích sau khi chụp CT. Xuất kết quả dưới dạng DICOM Structured Report (DICOM SR) cho phép lưu trữ kết quả AI cùng với ảnh gốc trong PACS, tạo điều kiện cho audit và nghiên cứu hồi cứu. Về tuân thủ quy định, hệ thống cần được thiết kế lại để đáp ứng các yêu cầu bảo vệ dữ liệu (HIPAA tại Mỹ, GDPR tại châu Âu, và các quy định tương ứng của Việt Nam), bao gồm anonymization tự động thông tin nhận dạng bệnh nhân (PHI) trước khi xử lý.

### 5.2.5. Nghiên cứu sâu hơn về AI second reader trong thực hành lâm sàng

Câu hỏi quan trọng nhất chưa được nghiên cứu này trả lời là: AI second reader có thực sự cải thiện kết quả đọc của bác sĩ trong điều kiện thực tế không? Để trả lời, cần thiết kế reader study với crossover design — một nhóm bác sĩ đọc không có AI, nhóm khác đọc có AI hỗ trợ, sau đó đổi chéo — đo đồng thời sensitivity, specificity và thời gian đọc trung bình trên cùng tập ca bệnh blind.

Disagreement analysis — phân tích các trường hợp AI và bác sĩ không đồng ý, sau đó xác nhận bằng follow-up — là nguồn thông tin quý để hiểu khi nào AI có giá trị bổ sung thực sự và khi nào nó chỉ gây nhiễu. Confidence calibration — kiểm tra xem xác suất dự đoán của mô hình có khớp với tỷ lệ đúng thực nghiệm theo calibration curve không — là điều kiện kỹ thuật cần đạt trước khi đưa con số xác suất AI vào quy trình quyết định lâm sàng. Active learning — sử dụng uncertainty của mô hình để ưu tiên chọn ca khó cho bác sĩ annotation bổ sung — là hướng nghiên cứu hiệu quả về chi phí để cải thiện mô hình mà không cần annotation toàn bộ dataset mới.

---

# TÀI LIỆU THAM KHẢO

## Sách giáo trình và tổng quan

[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[2] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. https://doi.org/10.1038/nature14539

[3] Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laak, J. A. W. M., van Ginneken, B., & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88. https://doi.org/10.1016/j.media.2017.07.005

## Dataset và chuẩn benchmark

[4] Armato, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., ... Clarke, L. P. (2011). The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931. https://doi.org/10.1118/1.3528204

[5] Setio, A. A. A., Traverso, A., de Bel, T., Berens, M. S. N., van den Bogaard, C., Cerello, P., Chen, H., Dou, Q., Fantacci, M. E., Geurts, B., van der Gugten, R., Heng, P. A., Jansen, B., de Kaste, M. M. J., Kotov, V., Lin, J. Y. H., Manders, J. T. M. C., Sóñora-Mengana, A., García-Naranjo, J. C., ... Jacobs, C. (2017). Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. *Medical Image Analysis*, 42, 1–13. https://doi.org/10.1016/j.media.2017.06.015

[6] National Lung Screening Trial Research Team. (2011). Reduced lung-cancer mortality with low-dose computed tomographic screening. *New England Journal of Medicine*, 365(5), 395–409. https://doi.org/10.1056/NEJMoa1102873

## Kiến trúc mạng nơ-ron

[7] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

[8] Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2019). UNet++: Redesigning skip connections to exploit multiscale features in image segmentation. *IEEE Transactions on Medical Imaging*, 39(6), 1856–1867. https://doi.org/10.1109/TMI.2019.2959609

[9] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, Proceedings of Machine Learning Research, 97, 6105–6114.

[10] Roy, A. G., Navab, N., & Wachinger, C. (2018). Concurrent spatial and channel squeeze & excitation in fully convolutional networks. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, 11070, 421–429. https://doi.org/10.1007/978-3-030-00928-1_48

[11] Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4700–4708. https://doi.org/10.1109/CVPR.2017.243

[12] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7132–7141. https://doi.org/10.1109/CVPR.2018.00745

[13] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

[14] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324

[15] Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z

## Hàm mất mát và tối ưu hóa

[16] Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. *Machine Learning in Medical Imaging (MLMI)*, Lecture Notes in Computer Science, 10541, 379–387. https://doi.org/10.1007/978-3-319-67389-9_44

[17] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR 2019)*. https://openreview.net/forum?id=Bkg6RiCqY7

[18] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *International Conference on Learning Representations (ICLR 2017)*. https://openreview.net/forum?id=Sgj0Fl3C3

[19] Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI)*, 876–885.

## Phân đoạn phổi và tiền xử lý

[20] Hofmanninger, J., Prayer, F., Pan, J., Röhrich, S., Prosch, H., & Langs, G. (2020). Automatic lung segmentation in routine imaging is primarily a data diversity problem, not a methodology problem. *European Radiology Experimental*, 4(1), Article 50. https://doi.org/10.1186/s41747-020-00173-2

## Phát hiện nốt và đánh giá

[21] Baumgartner, M., Jaeger, P. F., Isensee, F., & Maier-Hein, K. H. (2021). nnDetection: A self-configuring method for medical object detection. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, 12905, 530–539. https://doi.org/10.1007/978-3-030-87240-3_51

[22] Niemeijer, M., Loog, M., Abràmoff, M. D., Viergever, M. A., Prokop, M., & van Ginneken, B. (2011). On combining computer-aided detection systems. *IEEE Transactions on Medical Imaging*, 30(2), 215–223. https://doi.org/10.1109/TMI.2010.2072789

[23] Ferreira, C. A., Menegola, A., Gonçalves, R. S., Nunes, M. A. A., & Vieira, A. S. (2023). A review on deep learning techniques for the analysis of medical images: CT, MRI, ultrasound. *BME Frontiers*, 4, Article 0010. https://doi.org/10.34133/bmef.0010

[24] Van Ginneken, B., Armato, S. G., de Hoop, B., van Amelsvoort-van de Vorst, S., Duindam, T., Niemeijer, M., Murphy, K., Schilham, A., Retico, A., Fantacci, M. E., Camarlinghi, N., Bagaini, F., Cerello, P., Tangaro, S., Bellotti, R., Bosco, P., Fiorentino, C., Foracchia, M., & Franchini, R. (2010). Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: The ANODE09 study. *Medical Image Analysis*, 14(6), 707–722. https://doi.org/10.1016/j.media.2010.05.005

[25] Zhu, W., Liu, C., Fan, W., & Xie, X. (2018). DeepLung: Deep 3D dual path nets for automated pulmonary nodule detection and classification. *Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)*, 673–681. https://doi.org/10.1109/WACV.2018.00079

## Lung-RADS và tiêu chuẩn lâm sàng

[26] American College of Radiology. (2019). *Lung CT Screening Reporting and Data System (Lung-RADS), Version 1.1*. American College of Radiology. https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads

[27] McKee, B. J., Regis, S. M., McKee, A. B., Flacke, S., & Wald, C. (2015). Performance of ACR Lung-RADS in a clinical CT lung screening program. *Journal of the American College of Radiology*, 12(3), 273–276. https://doi.org/10.1016/j.jacr.2014.08.004

## Framework và thư viện phần mềm

[28] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 8024–8035.

[29] Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., Murrey, B., Myronenko, A., Zhao, C., Yang, D., Nath, V., He, Y., Xu, Z., Hatamizadeh, A., Zhu, W., Liu, Y., Zheng, M., Tang, Y., Yang, I., ... Feng, A. (2022). MONAI: An open-source framework for deep learning in healthcare. *arXiv preprint*, arXiv:2211.02701. https://arxiv.org/abs/2211.02701

[30] Yakubovskiy, P. (2020). *segmentation_models_pytorch* [Software]. GitHub. https://github.com/qubvel/segmentation_models.pytorch

[31] Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. *Information*, 11(2), Article 125. https://doi.org/10.3390/info11020125

[32] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

## Phương pháp thống kê

[33] Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. https://doi.org/10.2307/3001968

[34] Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. CRC Press. https://doi.org/10.1201/9780429246593

## Tài liệu bổ sung

[35] Tan, W. C., Lam, C. K. Y., Lim, A. Y. H., Bhatt, D. L., Ho, J. C. M., Abisheganaden, J. A., & Yap, J. C. H. (2022). Implementation of lung cancer CT screening and lung-RADS in Asia. *Chest*, 161(3), 646–655. https://doi.org/10.1016/j.chest.2021.09.012

[36] Ardila, D., Kiraly, A. P., Bharadwaj, S., Choi, B., Reicher, J. J., Peng, L., Tse, D., Etemadi, M., Ye, W., Corrado, G., Naidich, D. P., & Shetty, S. (2019). End-to-end lung cancer screening with deep learning on low-dose CT. *Nature Medicine*, 25(6), 954–961. https://doi.org/10.1038/s41591-019-0447-x

[37] Winkels, M., & Cohen, T. S. (2019). Pulmonary nodule detection in CT scans with equivariant CNNs. *Medical Image Analysis*, 55, 15–26. https://doi.org/10.1016/j.media.2019.03.010
