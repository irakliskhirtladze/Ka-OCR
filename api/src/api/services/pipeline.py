from dataclasses import dataclass
import numpy as np


@dataclass
class TextRegion:
    """Normalized OCR result for a single text region"""
    text: str
    bbox: tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class OCRItem:
    """OCR result for a single image"""
    image_id: str
    regions: list[TextRegion]

    def is_null(self) -> bool:
        """return true if ocr item regions list is empty"""
        return not self.regions


class OCRPipelineService:
    def __init__(self, image: np.ndarray):
        self.image = image

    async def pdf_to_images(self):
        pass

    async def func(self):
        pass
