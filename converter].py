from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    simplify=True,
    dynamic=False
)