from ultralytics import YOLO


def train() -> None:
    model = YOLO("yolo26n.pt")  # Load a pretrained 'nano' model
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        device=0  # GPU index
    )
