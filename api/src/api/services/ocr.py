import numpy as np
import torch
from PIL.Image import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


class OCRService:
    def __init__(self, processor: TrOCRProcessor, model: VisionEncoderDecoderModel):
        self.processor = processor
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def recognize(self, image: np.ndarray) -> str:
        """Takes an image as input and returns recognized text from it."""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()


class WordDetectorService:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    async def detect_words(self, image: np.ndarray) -> list[tuple]:
        """Takes an image as input and returns a list of word bounding box coordinates detected."""
        results = self.model(image, conf=0.25, verbose=False)[0]

        # extract bounding boxes: [x1, y1, x2, y2]
        boxes = results.boxes.xyxy.tolist()

        # Sort boxes: Top-to-Bottom, then Left-to-Right
        boxes.sort(key=lambda b: (b[1], b[0]))

        return boxes

