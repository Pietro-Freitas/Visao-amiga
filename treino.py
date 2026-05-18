from ultralytics import YOLO

model = YOLO("yolov8s-worldv2.pt")

model.train(
    data="modelo/data.yaml",
    epochs=20,
    imgsz=320,
    batch=4
)