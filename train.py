# Train lại YOLOv8n trên dataset projectdoan2 (cùng cấu hình với doxuanbach.ipynb)
# và lưu model vào models/best.pt để các tuần sau không phải train lại.
# Chạy: set biến môi trường ROBOFLOW_API_KEY rồi `.venv\Scripts\python train.py`
import os
import shutil
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO


def main():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    dataset = (rf.workspace("project2-snimr").project("projectdoan2").version(1)
               .download("yolov11", location="datasets/projectdoan2-1"))

    model = YOLO("yolov8n.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=400,
        imgsz=640,
        batch=16,
        patience=10,
        name="doan2_train_v8",
        project="runs/detect",
        exist_ok=True,
        plots=True,
        save=True,
        workers=2,  # Windows: để thấp tránh lỗi bộ nhớ dataloader
    )

    best = Path(model.trainer.best)
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(best, "models/best.pt")
    print("Đã lưu model:", Path("models/best.pt").resolve())


if __name__ == "__main__":
    main()
