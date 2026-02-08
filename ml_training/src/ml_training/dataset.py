import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor
import albumentations as A


class GeorgianTokenizer:
    """
    Custom tokenizer for Georgian alphabet.
    It tokenizes characters instead of words since fine-tuned model is supposed to recognize a single word.
    """
    def __init__(self, max_length: int = 32):
        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<s>"      # beginning of sequence
        self.eos_token = "</s>"     # end of sequence
        self.unk_token = "<unk>"    # unknown character

        # Georgian alphabet (33 letters) and other chars
        self.georgian_chars = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
        self.digits = "0123456789"
        self.roman = "IVXLCDM"
        self.punctuation = ".,-!?;:\"'()[]{}/=%+* &$#@~"

        # Build vocabulary: special tokens + Georgian characters
        self.vocab = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.vocab.extend(list(self.georgian_chars))
        self.vocab.extend(list(self.digits))
        self.vocab.extend(list(self.roman))
        self.vocab.extend(list(self.punctuation))

        # Create mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Token IDs for special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, padding: bool = True) -> list[int]:
        """Convert Georgian text to token IDs."""
        # Start with BOS token
        ids = [self.bos_token_id]

        # Convert each character
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_token_id))

        # Add EOS token
        ids.append(self.eos_token_id)

        # Truncate if too long
        if len(ids) > self.max_length:
            ids = ids[:self.max_length - 1] + [self.eos_token_id]

        # Pad if needed
        if padding:
            ids.extend([self.pad_token_id] * (self.max_length - len(ids)))

        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        chars = []
        for token_id in token_ids:
            if token_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            chars.append(self.id_to_char.get(token_id, ""))
        return "".join(chars)

    # def batch_decode(self, sequences: np.ndarray, skip_special_tokens: bool = True) -> list[str]:
    #     """Decodes a batch of token IDs (list of lists or torch Tensors)."""
    #     return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    # def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
    #     """Convert token IDs back to text, stopping at EOS."""
    #     # Convert torch tensor to list if necessary
    #     if hasattr(token_ids, "tolist"):
    #         token_ids = token_ids.tolist()
    #
    #     chars = []
    #     for token_id in token_ids:
    #         # Stop decoding if we hit the EOS token
    #         if token_id == self.eos_token_id and skip_special_tokens:
    #             break
    #
    #         # Skip BOS and PAD if requested
    #         if skip_special_tokens and token_id in (self.pad_token_id, self.bos_token_id):
    #             continue
    #
    #         chars.append(self.id_to_char.get(token_id, ""))
    #
    #     return "".join(chars)




class GeorgianOCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: str, processor: TrOCRProcessor, tokenizer: GeorgianTokenizer,
                 augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.tokenizer = tokenizer  # custom tokenizer
        self.augment = augment

        # augmentation pipeline
        self.aug_pipeline = A.Compose([
            A.Rotate(limit=(4, 4), border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                A.GaussNoise(p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GridDistortion(p=1.0),
                A.ElasticTransform(p=1.0),
                A.Perspective(scale=(0.02, 0.05), p=1.0),
            ], p=0.3),
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

        # Open and process image
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
        labels = self.tokenizer.encode(text)
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
