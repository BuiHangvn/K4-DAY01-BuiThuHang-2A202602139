# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU – Tesla T4

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- **Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):** `class_id=468`, `class_name="cab"`, `rank=1`, `score=0.510915`, `taxonomy_name="ImageNet-1K"`.
- **Record này mô tả toàn ảnh như thế nào?** Đây là prediction ở cấp ảnh: model gán `cab` là lớp có điểm cao nhất cho toàn bộ ảnh `traffic`. Record này không chỉ ra một chiếc xe cụ thể nằm ở đâu trong ảnh và cũng không tạo bounding box.
- **Ai định nghĩa class list mà checkpoint có thể dự đoán?** Class list được xác định bởi taxonomy/dataset mà checkpoint được huấn luyện để dự đoán. Với checkpoint này, taxonomy được ghi là `ImageNet-1K`.
- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?** `class_id` giúp máy xử lý nhất quán, `class_name` giúp con người đọc và diễn giải, còn `taxonomy_name` cho biết ID và tên lớp thuộc hệ phân loại nào. Cùng một ID có thể mang nghĩa khác nếu dùng taxonomy khác.
- **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?** Guideline cần quy định rõ tiêu chí chọn nhãn ở cấp ảnh, ví dụ ưu tiên chủ thể chính hay ngữ cảnh chính, và phải xác định bài toán là single-label hay multi-label để annotator xử lý nhất quán.
- **Vì sao model score không phải ground truth?** Score chỉ phản ánh mức độ tin cậy của model đối với prediction theo mô hình đã học. Model có thể dự đoán sai hoặc bị ảnh hưởng bởi dữ liệu huấn luyện, ngữ cảnh và chất lượng ảnh. Ground truth phải đến từ quy trình gán nhãn và kiểm tra theo guideline, không phải từ score của model.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- **Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):** `class_name="person"`, `score=0.912625`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68` pixel.
- **Diễn giải vị trí box bằng lời:** Với ảnh kích thước 640×427 pixel, box của `person` bắt đầu khoảng x=385, y=69 và kết thúc khoảng x=499, y=349. Vì vậy người này nằm chủ yếu ở nửa phải của ảnh và box kéo dài từ vùng phía trên xuống gần phía dưới ảnh.
- **So sánh số prediction ở hai threshold:** File evidence được lưu với `score_threshold=0.35`; tại threshold này sample `kitchen` có 11 prediction. Nếu lọc tiếp cùng các prediction này ở mức `score >= 0.50` thì còn 6 prediction.
- **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?** Threshold thấp hơn giữ lại nhiều prediction hơn nên tăng độ bao phủ nhưng reviewer phải kiểm tra nhiều trường hợp, trong đó có thể có nhiều false positive hơn. Threshold cao hơn giảm số prediction cần review nhưng có nguy cơ bỏ sót object có score thấp.
- **Đề xuất một quy tắc box chặt:** Bounding box nên là hình chữ nhật nhỏ nhất bao quanh phần nhìn thấy của object cần gán nhãn, hạn chế đưa background không cần thiết vào box và áp dụng quy tắc nhất quán cho mọi object cùng loại.
- **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?** Guideline cần quy định có gán nhãn object bị che khuất/cắt mép hay không, box chỉ bao phần nhìn thấy hay ước lượng toàn object, và mức che khuất nào phải bỏ qua hoặc chuyển reviewer/Lab Coach quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- **Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):** `instance_id="kitchen-001"`, `class_name="person"`, `score=0.899318`, `polygon_point_count=348`; một số điểm đầu của polygon là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
- **Polygon bổ sung chi tiết gì so với box?** Box chỉ mô tả một hình chữ nhật bao quanh object, trong khi polygon đi theo đường biên của object nên thể hiện hình dạng và vùng pixel thuộc object chi tiết hơn, đồng thời giảm phần background nằm trong vùng gán nhãn.
- **`instance_id` dùng để làm gì và không phải loại ID nào?** `instance_id` dùng để phân biệt từng instance riêng biệt trong prediction, ví dụ `kitchen-001` và `kitchen-002`. Nó không phải `class_id`, không phải `coco_image_id`, và cũng không phải định danh vĩnh viễn của người/vật ngoài đời thực.
- **Đề xuất một quy tắc biên mask:** Mask nên bám theo đường biên nhìn thấy của object, không lấy background, không tự mở rộng sang phần không quan sát được và phải xử lý biên theo cùng một quy tắc cho toàn bộ dataset.
- **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?** Guideline cần quy định cách xác định đường biên khi hai object tiếp xúc, khi biên bị mờ hoặc khi một phần object bị che. Nếu không thể xác định nhất quán từ ảnh và guideline thì cần chuyển reviewer/Lab Coach quyết định thay vì tự suy đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn ảnh theo taxonomy | Ảnh có nhiều chủ thể; khó xác định chủ thể/ngữ cảnh nào quyết định nhãn | Đọc guideline và chọn nhãn cấp ảnh theo đúng taxonomy | Kiểm tra nhãn có đúng taxonomy và đúng quy tắc single-label/multi-label |
| Phát hiện vật thể | Mỗi object có `class` + bounding box | Object nhỏ, che khuất, cắt mép, box có thể quá rộng/chật hoặc object bị bỏ sót | Gán đúng class và vẽ box chặt cho từng object đủ điều kiện | Kiểm tra class, object bị thiếu/trùng và độ chặt của box |
| Instance segmentation | Mỗi instance có `class` + mask/polygon | Biên mờ, object tiếp xúc/che khuất, polygon có thể ăn vào background hoặc thiếu vùng object | Tách từng instance và vẽ mask theo biên nhìn thấy theo guideline | Kiểm tra đường biên, vùng thừa/thiếu, tách instance và tính nhất quán |

## 5. An toàn dữ liệu

- **Một quy tắc bảo vệ dữ liệu:** Chỉ sử dụng dữ liệu được chương trình cung cấp, cho phép sử dụng và đúng phạm vi bài lab; không tự ý chia sẻ, tải lên dịch vụ bên ngoài hay bên thứ ba hoặc đưa dữ liệu cá nhân/nhạy cảm vào report và output.
- **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:** GV/Lab Coach phụ trách bài thực hành trước khi tiếp tục xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
