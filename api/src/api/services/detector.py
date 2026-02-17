from PIL.Image import Image
from ultralytics import YOLO


class WordDetectorService:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_words(self, image: Image) -> list[tuple]:
        """Takes an image as input and returns a list of word bounding box coordinates detected."""
        results = self.model(image, conf=0.25, verbose=False)[0]

        # extract bounding boxes: [x1, y1, x2, y2]
        boxes = results.boxes.xyxy.tolist()

        # Sort boxes: Top-to-Bottom, then Left-to-Right
        boxes.sort(key=lambda b: (b[1], b[0]))

        return boxes
