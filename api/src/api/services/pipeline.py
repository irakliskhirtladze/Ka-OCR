from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
import torch
from PIL import Image
from fastapi import UploadFile
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


@dataclass
class TextRegion:
    """Normalized OCR result for a single text region"""
    text: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ImageItem:
    """OCR result for a single image"""
    img_id: str
    img: np.ndarray
    regions: list[TextRegion]

    def is_null(self) -> bool:
        """return true if ocr item regions list is empty"""
        return not self.regions


class OCRPipelineService:
    def __init__(self,
                 yolo_model: YOLO,
                 trocr_model: VisionEncoderDecoderModel,
                 processor: TrOCRProcessor):
        self.detector = yolo_model
        self.recognizer = trocr_model
        self.processor = processor

    def detect_bboxes(self, img_item: ImageItem) -> ImageItem:
        # Run inference
        results = self.detector.predict(img_item.img, conf=0.25, device=self.detector.device, verbose=False)

        # Extract the boxes from the first result
        result = results[0]

        # YOLO26 boxes are in 'xyxy' format: [x1, y1, x2, y2]
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            # Create a TextRegion (leaving 'text' empty for the next step)
            region = TextRegion(
                text="",
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )
            img_item.regions.append(region)

        return img_item


if __name__ == "__main__":
    from api.core.config import settings
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from ultralytics import YOLO
    import torch

    print(f"Initializing models on {settings.DEVICE}...")

    # TrOCR Loading
    processor = TrOCRProcessor.from_pretrained(settings.TROCR_MODEL_ID)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(settings.TROCR_MODEL_ID)
    trocr_model.to(settings.DEVICE)

    # YOLO Loading - pass the path inside the parentheses!
    yolo_model = YOLO(settings.YOLO_MODEL, task="detect")

    print("Pipeline ready for testing.")

    img_path = str(settings.BASE_DIR / "sample_files/1.jpg")
    image = cv2.imread(img_path)
    img_item = ImageItem(
        img_id=img_path,
        img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        regions=[],
    )
    pipeline = OCRPipelineService(yolo_model, trocr_model, processor)
    img_item_with_bboxes = pipeline.detect_bboxes(img_item)

    # Draw bboxes on image with opencv
    for region in img_item_with_bboxes.regions:
        cv2.rectangle(img_item_with_bboxes.img, (region.x1, region.y1), (region.x2, region.y2), (255, 0, 0), 2)

    Image.fromarray(img_item_with_bboxes.img).show()
