"""Module theo dõi phương tiện: YOLO + ByteTrack, gán Tracking ID ổn định cho từng xe.

Dùng chung cho nhóm:
- Đếm xe qua vạch ảo: dùng `track.track_id` và `track.anchor` (điểm giữa cạnh dưới box).
- Luồng video camera: gọi `tracker.update(frame)` cho từng frame đọc từ cv2.VideoCapture.
- Model ONNX: truyền đường dẫn file .onnx vào `weights`.

Ví dụ:
    tracker = VehicleTracker()
    cap = cv2.VideoCapture("video.mp4")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for t in tracker.update(frame):
            print(t.track_id, t.cls_name, t.box)

Chạy nhanh từ dòng lệnh (xuất video có vẽ ID):
    python vehicle_tracker.py videos/abc.mp4 --out output/abc_tracked.mp4
"""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11s_best.pt"  # model được chọn ở tuần 7
DEFAULT_TRACKER = HERE / "trackers" / "bytetrack_tuned.yaml"  # cấu hình chọn ở notebook tuần 8


@dataclass
class Track:
    track_id: int
    cls: int  # lớp đã làm mượt theo lịch sử của ID (bỏ phiếu đa số)
    cls_name: str
    raw_cls: int  # lớp model dự đoán ở frame hiện tại
    conf: float
    box: tuple  # (x1, y1, x2, y2) theo pixel

    @property
    def anchor(self):
        # điểm giữa cạnh dưới box, tức vị trí bánh xe chạm đường: dùng để xét xe đi qua vạch ảo
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


class VehicleTracker:
    def __init__(self, weights=DEFAULT_WEIGHTS, tracker=DEFAULT_TRACKER, conf=0.1, iou=0.7, imgsz=640,
                 device=None, class_window=30):
        """
        conf: ngưỡng confidence đưa vào tracker. Để thấp (0.1) vì ByteTrack dùng cả detection điểm thấp
              để nối lại xe bị che; việc lọc do track_high_thresh / new_track_thresh trong file cấu hình.
        iou: ngưỡng IoU của NMS.
        class_window: số frame gần nhất dùng để bỏ phiếu lớp cho mỗi ID, tránh một xe lúc là
              'truck' lúc là 'car'.
        """
        tracker = Path(tracker)
        if not tracker.exists():  # chưa chạy notebook tuần 8 thì dùng cấu hình mặc định của Ultralytics
            tracker = Path("bytetrack.yaml")
        self.weights = str(weights)
        self.track_args = dict(tracker=str(tracker), conf=conf, iou=iou, imgsz=imgsz, device=device,
                               persist=True, verbose=False)
        self.class_window = class_window
        self.reset()

    def reset(self):
        """Gọi khi chuyển sang video khác để ID đánh số lại từ đầu."""
        self.model = YOLO(self.weights, task="detect")
        self.names = self.model.names
        self._cls_hist = defaultdict(lambda: deque(maxlen=self.class_window))

    def update(self, frame):
        """Nhận một frame BGR (numpy), trả về danh sách Track của các xe đang được theo dõi."""
        boxes = self.model.track(frame, **self.track_args)[0].boxes
        if boxes is None or boxes.id is None:
            return []
        tracks = []
        for tid, c, s, box in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(),
                                  boxes.conf.tolist(), boxes.xyxy.tolist()):
            hist = self._cls_hist[tid]
            hist.append(c)
            cls = Counter(hist).most_common(1)[0][0]
            tracks.append(Track(tid, cls, self.names[cls], c, s, tuple(box)))
        return tracks


def id_color(track_id):
    # mỗi ID một màu cố định, dễ nhìn ra khi ID bị đổi
    h = (track_id * 47) % 180
    hsv = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in hsv)


def draw_tracks(frame, tracks, trails=None):
    """Vẽ box, nhãn '#ID lớp' và vệt di chuyển (nếu truyền trails: dict id -> deque các anchor)."""
    side = max(frame.shape[:2])
    t = max(1, round(side / 900))  # độ dày nét
    scale = side / 2400  # cỡ chữ: khoảng 0.4 với video 960px, 0.8 với video 1920px
    for tr in tracks:
        color = id_color(tr.track_id)
        x1, y1, x2, y2 = map(int, tr.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, t)
        label = f"#{tr.track_id} {tr.cls_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)
        if trails is not None:
            trails[tr.track_id].append(tuple(map(int, tr.anchor)))
            pts = list(trails[tr.track_id])
            for a, b in zip(pts, pts[1:]):
                cv2.line(frame, a, b, color, t)
    return frame


def open_writer(path, fps, size):
    """Mở VideoWriter H.264 (xem được trong VS Code / trình duyệt); máy không hỗ trợ thì dùng mp4v."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Không mở được VideoWriter cho {path}")


def main():
    ap = argparse.ArgumentParser(description="Theo dõi xe trong video bằng YOLO + ByteTrack")
    ap.add_argument("video")
    ap.add_argument("--out", help="đường dẫn video kết quả (.mp4)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    args = ap.parse_args()

    tracker = VehicleTracker(args.weights, args.tracker)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(args.out, fps, (w, h)) if args.out else None
    trails = defaultdict(lambda: deque(maxlen=30))
    ids = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.update(frame)
        ids.update(t.track_id for t in tracks)
        if writer:
            writer.write(draw_tracks(frame, tracks, trails))
    cap.release()
    if writer:
        writer.release()
    print(f"Đã theo dõi {len(ids)} ID", f"-> {args.out}" if args.out else "")


if __name__ == "__main__":
    main()
