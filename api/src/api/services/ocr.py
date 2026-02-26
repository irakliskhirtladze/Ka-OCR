from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


@dataclass
class TextRegion:
    """Normalized OCR result for a single text region."""
    text: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ImageItem:
    """OCR result for a single image."""
    img_id: str
    img: np.ndarray
    regions: list[TextRegion] = field(default_factory=list)

    def is_null(self) -> bool:
        """Returns True if no text regions were detected."""
        return not self.regions


class OCR:
    def __init__(
        self,
        yolo_model: YOLO,
        trocr_model: torch.nn.Module,
        processor: TrOCRProcessor,
        device: str,
    ):
        self.detector = yolo_model
        self.recognizer = trocr_model
        self.processor = processor
        self.device = device

    def detect_bboxes(self, img_item: ImageItem) -> ImageItem:
        """Runs YOLO word detection and populates img_item.regions with bbox coordinates."""
        results = self.detector.predict(img_item.img, conf=0.25, device=self.device, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            img_item.regions.append(TextRegion(text="", x1=x1, y1=y1, x2=x2, y2=y2))
        img_item.regions.sort(key=lambda r: (r.y1, r.x1))
        return img_item

    @staticmethod
    def crop_region(img: np.ndarray, region: TextRegion) -> np.ndarray:
        """Crops a single text region from an image using its bounding box."""
        return img[region.y1:region.y2, region.x1:region.x2]

    async def recognize(self, image: np.ndarray) -> str:
        """Runs TrOCR on a cropped region image and returns recognized text."""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.recognizer.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True,
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    async def process_image(self, img_item: ImageItem) -> ImageItem:
        """Full OCR pipeline for one image: detect bboxes → crop regions → recognize text."""
        self.detect_bboxes(img_item)
        for region in img_item.regions:
            cropped = self.crop_region(img_item.img, region)
            region.text = await self.recognize(cropped)
        return img_item
