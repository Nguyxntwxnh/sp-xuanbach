# BÁO CÁO TUẦN 8 – TÍCH HỢP BYTETRACK VỚI YOLO ĐỂ GÁN TRACKING ID

- Người thực hiện: Đỗ Xuân Bách
- Ngày: 01/11/2026
- Model nhận diện: YOLO11s (yolo11s_best.pt), chọn theo đề xuất của báo cáo tuần 7
- Thuật toán theo dõi: ByteTrack (Ultralytics 8.4.197)
- Phần cứng: GPU NVIDIA GeForce RTX 3050 Laptop GPU

## 1. Mục tiêu

Tích hợp thuật toán theo dõi đa đối tượng ByteTrack với model YOLO để mỗi xe trong video được gán một Tracking ID duy nhất và giữ nguyên ID đó trong suốt thời gian xuất hiện, kể cả khi bị xe khác che tạm thời. ID ổn định là điều kiện để module đếm xe qua vạch ảo không đếm trùng hoặc đếm sót.

## 2. Phương pháp

### 2.1. ByteTrack

- Mỗi frame, YOLO phát hiện xe; ByteTrack dùng bộ lọc Kalman dự đoán vị trí mới của các track đang có.
- Bước 1: các detection điểm cao (≥ track_high_thresh = 0.25) được ghép với track theo IoU.
- Bước 2: các detection điểm thấp (≥ track_low_thresh = 0.1) được ghép với các track còn lại. Đây là điểm khác biệt của ByteTrack: xe bị che một phần thường có điểm thấp, nhưng vẫn được nối vào đúng ID thay vì bị bỏ.
- Detection điểm cao không ghép được (≥ new_track_thresh) tạo ID mới; track không được ghép sẽ được giữ track_buffer frame trước khi xóa, để nối lại nếu xe xuất hiện trở lại.

### 2.2. Tích hợp

- Module vehicle_tracker.py bọc model YOLO + ByteTrack thành lớp VehicleTracker: mỗi lần gọi update(frame) trả về danh sách xe gồm track_id, lớp, confidence, box và điểm neo (giữa cạnh dưới box).
- Detection được đưa vào tracker ở ngưỡng confidence 0.1 để ByteTrack có detection điểm thấp cho bước 2.
- Làm mượt lớp: lớp của mỗi ID lấy theo đa số trong 30 frame gần nhất, tránh một xe lúc là xe tải lúc là ô tô (lỗi nhầm lớp đã thấy ở tuần 6), giúp đếm xe theo loại chính xác hơn.

### 2.3. Cách đánh giá

Video không có nhãn tracking chuẩn nên dùng các chỉ số gián tiếp: số ID, độ dài track trung vị, tỉ lệ track vụn (ID tồn tại < 0.5 giây), số ca nghi đổi ID (ID biến mất giữa khung hình và trong 2.0 giây có ID mới cùng lớp xuất hiện ngay gần đó – một xe bị gán nhiều ID), số lần nhảy vị trí bất thường (tâm box của một ID dịch xa hơn kích thước box giữa 2 lần xuất hiện – một ID bị ghép sang xe khác), tỉ lệ ID bị đổi lớp và tốc độ xử lý. Cấu hình được chọn theo tổng hai loại lỗi ID trên 100 xe ước lượng, vì giảm lỗi loại này bằng cách nới lỏng ghép cặp thường làm tăng lỗi loại kia. Bốn cấu hình ByteTrack được so sánh trên tối đa 60 giây đầu của mỗi video, sau đó cấu hình tốt nhất được chạy trên toàn bộ video và kiểm tra bằng mắt qua video kết quả.

## 3. Dữ liệu

**Bảng 1. Video dùng để đánh giá**

| Video | Độ phân giải | FPS | Số frame | Thời lượng (giây) |
|---|---|---|---|---|
| FREE STOCK FOOTAGE - Heavy traffic.mp4 | 1920×1080 | 25 | 283 | 11.3 |

## 4. So sánh các cấu hình ByteTrack

**Bảng 2. Tham số các cấu hình**

| Cấu hình | track_high_thresh | track_low_thresh | new_track_thresh | track_buffer | match_thresh | fuse_score |
|---|---|---|---|---|---|---|
| A. Mặc định | 0.25 | 0.1 | 0.25 | 30 | 0.8 | True |
| B. Giữ ID lâu hơn | 0.25 | 0.1 | 0.25 | 60 | 0.8 | True |
| C. B + hạn chế ID mới | 0.25 | 0.1 | 0.4 | 60 | 0.8 | True |
| D. C + ghép cặp nới lỏng | 0.25 | 0.1 | 0.4 | 90 | 0.9 | True |

**Bảng 3. Kết quả các cấu hình (gộp 1 video, tối đa 60 giây đầu mỗi video)**

| Cấu hình | Số ID | Độ dài track trung vị (giây) | Track vụn (%) | Nghi đổi ID | Nghi đổi ID / 100 ID | Nhảy vị trí bất thường | Nhảy bất thường / 100 ID | ID bị đổi lớp – trước làm mượt (%) | ID bị đổi lớp – sau làm mượt (%) | FPS xử lý | Lỗi ID / 100 xe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Mặc định | 586 | 0.7 | 42.7 | 358 | 61.1 | 8 | 1.4 | 11.1 | 3.4 | 34.8 | 165.6 |
| B. Giữ ID lâu hơn | 563 | 0.8 | 40.7 | 335 | 59.5 | 12 | 2.1 | 11 | 4.1 | 30.4 | 157 |
| C. B + hạn chế ID mới | 342 | 1.12 | 33.3 | 172 | 50.3 | 14 | 4.1 | 13.7 | 3.5 | 31.9 | 84.2 |
| D. C + ghép cặp nới lỏng | 221 | 3.88 | 12.2 | 43 | 19.5 | 29 | 13.1 | 32.1 | 10.4 | 31.3 | 32.6 |

![Hình 1. So sánh các cấu hình ByteTrack](<output/so_sanh_cau_hinh.png>)

*Hình 1. So sánh các cấu hình ByteTrack*

- Cấu hình được chọn: D. C + ghép cặp nới lỏng (tổng lỗi ID ít nhất). So với mặc định: lỗi ID / 100 xe 165.6 → 32.6, nghi đổi ID 358 → 43, nhảy vị trí bất thường 8 → 29, track vụn 42.7 → 12.2 %, số ID 586 → 221, độ dài track trung vị 0.7 → 3.88 giây.
- Làm mượt lớp giảm tỉ lệ ID bị đổi lớp từ 32.1% xuống 10.4%.

## 5. Kết quả với cấu hình được chọn

**Bảng 4. Kết quả trên toàn bộ video**

| Video | Số ID | Độ dài track trung vị (giây) | Track vụn (%) | Nghi đổi ID | Nghi đổi ID / 100 ID | Nhảy vị trí bất thường | Nhảy bất thường / 100 ID | ID bị đổi lớp – trước làm mượt (%) | ID bị đổi lớp – sau làm mượt (%) | FPS xử lý |
|---|---|---|---|---|---|---|---|---|---|---|
| FREE STOCK FOOTAGE - Heavy traffic.mp4 | 221 | 3.88 | 12.2 | 43 | 19.5 | 29 | 13.1 | 32.1 | 10.4 | 31 |

- Video kết quả: [FREE STOCK FOOTAGE - Heavy traffic_tracked.mp4](<output/FREE STOCK FOOTAGE - Heavy traffic_tracked.mp4>)

![Hình 2. Dòng thời gian các ID – FREE STOCK FOOTAGE - Heavy traffic.mp4](<output/dong_thoi_gian_id.png>)

*Hình 2. Dòng thời gian các ID – FREE STOCK FOOTAGE - Heavy traffic.mp4*

![Hình 3. Phân bố độ dài track: mặc định và cấu hình được chọn](<output/do_dai_track.png>)

*Hình 3. Phân bố độ dài track: mặc định và cấu hình được chọn*

![Hình 4. Frame 70 – FREE STOCK FOOTAGE - Heavy traffic](<output/snapshots/FREE STOCK FOOTAGE - Heavy traffic_f70.jpg>)

*Hình 4. Frame 70 – FREE STOCK FOOTAGE - Heavy traffic*

![Hình 5. Frame 141 – FREE STOCK FOOTAGE - Heavy traffic](<output/snapshots/FREE STOCK FOOTAGE - Heavy traffic_f141.jpg>)

*Hình 5. Frame 141 – FREE STOCK FOOTAGE - Heavy traffic*

![Hình 6. Frame 212 – FREE STOCK FOOTAGE - Heavy traffic](<output/snapshots/FREE STOCK FOOTAGE - Heavy traffic_f212.jpg>)

*Hình 6. Frame 212 – FREE STOCK FOOTAGE - Heavy traffic*

**Quan sát từ video:**

- Phần lớn ô tô ở giữa khung hình giữ nguyên ID suốt video: ví dụ #1, #2, #3, #4, #55, #59, #65, #101, #119, #124, #163, #166, #171 giữ cùng ID từ frame 70 đến frame 141 (gần 3 giây); vệt di chuyển ngắn và liền mạch, đúng với cảnh xe nhích chậm khi kẹt xe.
- Vẫn có ca ID bị ghép nhầm sang xe khác: vệt di chuyển xuất hiện các đoạn thẳng dài vắt ngang khung hình, trong khi xe trong cảnh kẹt không thể di chuyển xa như vậy. Đây là cái giá của cấu hình D (ghép cặp nới lỏng): số lần nhảy vị trí bất thường tăng từ 8 (mặc định) lên 29, đổi lại số ca nghi đổi ID giảm từ 358 xuống 43.
- Một số box lớn trùm 2–4 xe (vd. #150, #185, #215). Đây là lỗi của bước nhận diện; tracker giữ một ID cho cả cụm, nên module đếm xe có thể đếm thiếu ở những chỗ này.
- Xe buýt thường bị gán nhãn car/truck (vd. #7 lúc là car, lúc là truck) và nhiều xe máy ở hàng đầu không được nhận diện, khớp với kết quả tuần 6 và 7 (lớp bus ít dữ liệu, xe máy bị che khuất). Đây là giới hạn của model nhận diện, ByteTrack không khắc phục được.
- Video chỉ dài 11 giây và được dùng cho cả bước chọn cấu hình lẫn đánh giá, nên kết quả có thể lạc quan; cần thêm video khác để kiểm tra lại cấu hình đã chọn.
- Đã thử thêm một video đường cao tốc phủ tuyết ở Bắc Mỹ (camera rất cao, xe nhỏ, có viền đen hai bên): model của dự án gần như không nhận diện được xe mà sinh box giả chiếm 42–46% khung hình, nên video này bị loại khỏi đánh giá (để ở videos/khong_phu_hop/). Model YOLO11s gốc (train trên COCO) vẫn nhận diện được xe trên video đó, cho thấy model của nhóm (train trên 135 ảnh) chưa khái quát được sang cảnh khác xa dữ liệu train.

## 6. Hướng dẫn tích hợp cho nhóm

Module nằm ở tuan8/vehicle_tracker.py, tự dùng model models/yolo11s_best.pt và cấu hình tuan8/trackers/bytetrack_tuned.yaml.

**Luồng video camera (Phạm Văn Tường) – đọc frame bằng OpenCV và gọi update():**

```python
import cv2
from vehicle_tracker import VehicleTracker, draw_tracks

tracker = VehicleTracker()
cap = cv2.VideoCapture(0)  # hoặc đường dẫn video / RTSP
while True:
    ok, frame = cap.read()
    if not ok:
        break
    tracks = tracker.update(frame)
    cv2.imshow("tracking", draw_tracks(frame, tracks))
    if cv2.waitKey(1) == 27:
        break
```

**Đếm xe qua vạch ảo (Nguyễn Thành Đạt) – dùng track_id và điểm neo để mỗi xe chỉ đếm một lần:**

```python
from collections import Counter

LINE_Y = 400                 # vạch ngang ở y = 400 px
last_y, counted, counts = {}, set(), Counter()
# đoạn dưới đặt trong vòng lặp đọc frame
for t in tracker.update(frame):
    x, y = t.anchor          # điểm giữa cạnh dưới box
    prev = last_y.get(t.track_id)
    if prev is not None and prev < LINE_Y <= y and t.track_id not in counted:
        counted.add(t.track_id)
        counts[t.cls_name] += 1   # t.cls_name đã được làm mượt theo lịch sử ID
    last_y[t.track_id] = y
```

**Model ONNX (Nguyễn Vũ Tùng) – chỉ cần đổi đường dẫn model:**

```python
tracker = VehicleTracker(weights="../models/yolo11s_best.onnx")
```

**Chạy nhanh từ dòng lệnh để xuất video có vẽ ID:**

```bash
python vehicle_tracker.py videos/ten_video.mp4 --out output/ten_video_tracked.mp4
```

## 7. Hạn chế và lưu ý

### 7.1. Hạn chế

- Không có nhãn tracking chuẩn nên chưa tính được các chỉ số chuẩn như MOTA, IDF1, HOTA; các chỉ số gián tiếp chỉ phản ánh xu hướng.
- Chỉ số "nghi đổi ID" có thể đếm nhầm khi hai xe cùng lớp đi sát nhau; chỉ số "nhảy vị trí bất thường" chỉ bắt được ca gộp nhầm khi hai xe cách xa nhau, không bắt được khi ID chuyển sang xe ngay bên cạnh. Vì vậy vẫn cần xem video kết quả để kiểm tra.
- Kết quả phụ thuộc video dùng để đánh giá (1 video); góc quay, mật độ xe và độ phân giải khác có thể cần tinh chỉnh lại.
- ByteTrack chỉ dùng vị trí và IoU, không dùng đặc trưng hình ảnh, nên xe bị che hoàn toàn lâu hơn track_buffer sẽ bị gán ID mới khi xuất hiện lại.

### 7.2. Lưu ý khi đọc và tái lập kết quả

- FPS xử lý đo cho cả nhận diện và tracking, chưa gồm đọc/ghi video; dao động giữa các lần chạy như đã ghi ở báo cáo tuần 7.
- Video kết quả ghi bằng codec H.264 (avc1) nếu máy hỗ trợ, nếu không thì mp4v; nếu trình xem không mở được, dùng VLC.
- Phần "Quan sát từ video" (nếu có) được viết tay sau khi xem video kết quả; đổi video hoặc model thì cần cập nhật.
- Cách chạy lại: chép video vào tuan8/videos/, mở tuan8/tuần 8.ipynb, chọn kernel .venv của dự án rồi Run all. Notebook ghi đè trackers/bytetrack_tuned.yaml bằng cấu hình tốt nhất cho bộ video hiện có.

## Phụ lục: file đi kèm

- vehicle_tracker.py – module theo dõi xe
- trackers/bytetrack_tuned.yaml – cấu hình được chọn; bytetrack_a…d.yaml – các cấu hình thử
- output/do_dai_track.png
- output/dong_thoi_gian_id.png
- output/FREE STOCK FOOTAGE - Heavy traffic_tracked.mp4
- output/FREE STOCK FOOTAGE - Heavy traffic_tracks.csv
- output/ket_qua_cau_hinh_chon.csv
- output/so_sanh_cau_hinh.csv
- output/so_sanh_cau_hinh.png
- output/so_sanh_cau_hinh_theo_video.csv
- output/video.csv
- output/snapshots/ – ảnh chụp frame có vẽ ID
