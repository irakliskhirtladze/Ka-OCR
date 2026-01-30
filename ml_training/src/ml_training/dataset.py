import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

from ml_training.setup import Paths


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

        # Georgian alphabet (33 letters)
        self.georgian_chars = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"

        # Build vocabulary: special tokens + Georgian characters
        self.vocab = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.vocab.extend(list(self.georgian_chars))

        # Create mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Token IDs for special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.max_length = max_length

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

    def __len__(self) -> int:
        return len(self.vocab)


class GeorgianOCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: str, processor, tokenizer: GeorgianTokenizer):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.tokenizer = tokenizer  # custom tokenizer

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

        # Use Processor for Normalization
        pixel_values = self.processor(new_img, return_tensors="pt").pixel_values

        # Tokenize Georgian Text
        labels = self.tokenizer.encode(text)

        # Replace padding token id with -100 so it's ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }


def train_test(paths: Paths, test_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(paths.dataset_dir/"metadata.csv")
    return train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )
