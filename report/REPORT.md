# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** Chưa có thông tin Python và PyTorch trong output / Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không thay đổi; detection và segmentation dùng threshold `0.35`.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

`class_id=468`, `class_name="cab"`, `rank=1`, `score=0.510915`, `taxonomy_name="ImageNet-1K"`.

- Record này mô tả toàn ảnh như thế nào?

Record mô tả toàn bộ ảnh, không mô tả riêng một xe. Ảnh có nhiều xe buýt, ô tô và người nhưng model xếp hạng một lớp cấp ảnh; vì vậy `cab` chỉ là prediction có score cao nhất, không phải nhãn chuẩn.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

Class list do taxonomy ImageNet-1K gắn với checkpoint `yolo11n-cls.pt` định nghĩa, không phải do model tự nghĩ ra.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

`class_id` giúp máy xử lý ổn định, `class_name` giúp con người đọc hiểu, còn `taxonomy_name` cho biết ID và tên lớp thuộc hệ phân loại nào để tránh nhầm lẫn.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

Guideline cần quy định dùng một nhãn hay nhiều nhãn, cách chọn chủ thể chính và cách xử lý khi nhiều chủ thể cùng nổi bật.

- Vì sao model score không phải ground truth?

Model score chỉ thể hiện mức tự tin của model đối với prediction. Ground truth phải được con người xác nhận theo guideline vì model có thể dự đoán sai hoặc bỏ sót.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

`class_name="person"`, `score=0.912624`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`, `bbox_width=113.58`, `bbox_height=279.68` pixel.

- Diễn giải vị trí box bằng lời:

Box bao quanh người đứng ở nửa bên phải ảnh, từ điểm trái-trên `(385.33, 69.24)` đến điểm phải-dưới `(498.92, 348.92)`. Gốc tọa độ nằm ở góc trên bên trái ảnh.

- So sánh số prediction ở hai threshold:

Với sample `kitchen`, threshold `0.35` giữ 11 prediction. Khi lọc ở threshold `0.60` thì còn 6 prediction. Notebook còn thử threshold `0.20`, nhưng kết quả của mức này không được lưu trong output.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

Hạ threshold thường giữ nhiều prediction hơn, tăng độ bao phủ nhưng reviewer phải kiểm tra nhiều false positive hơn. Nâng threshold giảm số prediction cần xem nhưng có nguy cơ bỏ sót object.

- Đề xuất một quy tắc box chặt:

Mỗi box chỉ bao đúng một object và bám sát phần nhìn thấy ngoài cùng của object, không chứa quá nhiều nền và không lấn sang object bên cạnh.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

Guideline cần quy định vẽ box quanh phần nhìn thấy hay ước lượng toàn object, tỷ lệ nhìn thấy tối thiểu để gán nhãn và trường hợp nào phải chuyển reviewer quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

`instance_id="kitchen-001"`, `class_name="person"`, `score=0.899318`, số điểm `348`; một phần `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], ...]`.

- Polygon bổ sung chi tiết gì so với box?

Polygon bám theo hình dạng và đường biên thực của object, nhờ đó tách object khỏi phần nền nằm bên trong hình chữ nhật của box.

- `instance_id` dùng để làm gì và không phải loại ID nào?

`instance_id` dùng để phân biệt từng object riêng trong output, kể cả khi chúng cùng lớp. Nó không phải class ID và không phải tracking ID theo dõi object qua nhiều frame.

- Đề xuất một quy tắc biên mask:

Mask phải bám sát phần biên nhìn thấy của đúng một instance, không lấy nền rõ ràng, không gộp hai object tiếp xúc và không tự vẽ phần bị che nếu guideline không yêu cầu.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

Guideline cần quy định độ sai lệch biên cho phép, cách tách các object tiếp xúc và có gán phần bị che hay không. Nếu không xác định được biên theo quy tắc thì annotator phải gắn cờ để reviewer xử lý.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn hoặc tập nhãn cho toàn ảnh theo taxonomy | Ảnh có nhiều phương tiện nhưng top-1 chỉ là `cab` | Chọn nhãn theo quy tắc chủ thể chính hoặc multi-label | Kiểm tra đúng taxonomy và tính nhất quán giữa các ảnh |
| Phát hiện vật thể | Một class và box `xyxy` cho mỗi object | Người bị cắt mép, nhiều bowl chồng nhau và một số vật thể bị bỏ sót | Tìm đủ object và vẽ box chặt cho từng instance | Kiểm tra thiếu/trùng object, sai class và box quá rộng hoặc quá chật |
| Instance segmentation | Một class, instance ID và polygon/mask cho mỗi object | Biên bàn phức tạp, spoon mảnh, object tiếp xúc hoặc bị che | Tách từng instance và vẽ mask bám biên | Kiểm tra mask ăn nền, thiếu vùng hoặc gộp/tách sai instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:

Chỉ sử dụng ảnh công khai được bài lab cung cấp; không upload ảnh cá nhân, dữ liệu khách hàng, dữ liệu nội bộ, mật khẩu hoặc token lên Colab/GitHub công khai.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

Tôi sẽ dừng xử lý, không upload hoặc commit dữ liệu đó và báo cho Lab Coach/GV qua kênh chính thức.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS` (không có output được lưu để xác nhận).
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
