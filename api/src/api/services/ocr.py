import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class OCRService:
    def __init__(self, processor: TrOCRProcessor, model: VisionEncoderDecoderModel):
        self.processor = processor
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def recognize(self, image: Image.Image) -> str:
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
