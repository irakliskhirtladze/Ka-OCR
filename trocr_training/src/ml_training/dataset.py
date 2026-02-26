import cv2
import numpy as np
import pandas as pd
import torch
import unicodedata
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
from transformers import PreTrainedTokenizerBase
import albumentations as A


class GeorgianOCRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_dir: str,
        processor: TrOCRProcessor,
        tokenizer: PreTrainedTokenizerBase,
        augment: bool = False,
        max_target_length: int = 32,
    ):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.augment = augment
        self.max_target_length = max_target_length

        self.aug_pipeline = A.Compose([
            A.Rotate(
                limit=(-4, 4),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                interpolation=cv2.INTER_LINEAR,
                p=0.7,
            ),
            A.GaussNoise(
                std_range=(0.1, 0.2),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.7,
            ),
            A.InvertImg(p=0.01),
            A.GaussianBlur(
                blur_limit=(7, 7),
                sigma_limit=(0.5, 0.5),
                p=0.7,
            ),
            A.ElasticTransform(
                alpha=1.0,
                sigma=5.0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                interpolation=cv2.INTER_LINEAR,
                p=0.7,
            ),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Performs image processing suitable for TrOCR model.
        Normalizes image tensor using TrOCR processor.
        uses custom tokenizer to encode label (text) characters.
        returns dict of image tensors and tokenized labels.
        """
        img_path = f"{self.dataset_dir}/{self.df.iloc[idx]['file_name']}"
        text = self.df.iloc[idx]['text']  # the text written on the image file
        text = unicodedata.normalize('NFC', str(text))

        # Open and process image0
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        target_size = 384

        # Scale height to target_size, width proportionally
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad to square
        new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
        new_img.paste(img, offset)

        # augment
        if self.augment:
            img_np = np.array(new_img)
            augmented = self.aug_pipeline(image=img_np)["image"]
            new_img = Image.fromarray(augmented)

        # Use Processor for Normalization
        pixel_values = self.processor(new_img, return_tensors="pt").pixel_values
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels
        }
