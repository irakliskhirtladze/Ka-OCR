import os
from pathlib import Path

import torch
from torchvision.utils import save_image
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ml_training.dataset import GeorgianTokenizer
from ml_training.setup import Paths


def save_debug_samples(dataloader, tokenizer, output_dir, n_images=200):
    """Saves un-normalized augmented samples by iterating through batches."""
    debug_path = os.path.join(output_dir, "debug_augs")
    os.makedirs(debug_path, exist_ok=True)

    count = 0

    # Loop through batches until we hit n_images
    for batch in dataloader:
        if count >= n_images:
            break

        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        # Process each image in the current batch
        for i in range(pixel_values.shape[0]):
            if count >= n_images:
                break

            # 1. Un-normalize
            img_tensor = pixel_values[i]
            img_min, img_max = img_tensor.min(), img_tensor.max()
            img_tensor = (img_tensor - img_min) / (img_max - img_min + 1e-5)  # Added epsilon to prevent div by zero

            # 2. Get label
            clean_label = labels[i][labels[i] != -100]
            decoded_text = tokenizer.decode(clean_label.tolist())

            # Sanitize filename (Georgian characters can be tricky in some filesystems)
            # We'll use a simple counter + sanitized text
            safe_text = "".join([c for c in decoded_text if c.isalnum() or c in (' ', '_')]).strip()[:20]
            fname = f"sample_{count}_{safe_text}.png"

            # 3. Save
            save_image(img_tensor, os.path.join(debug_path, fname))
            count += 1

    print(f"--- Successfully saved {count} debug images to: {debug_path} ---")


def test_against_real_images(paths: Paths, model, processor, tokenizer):
    print("Testing against real images...")
    ka_model_path = paths.drive_output_dir / "best_model.pt"  # The path to your .pt file
    sample_imgs_dir = Path("/content/drive/MyDrive/Colab Notebooks/trocr-ka/data/")
    if not sample_imgs_dir.exists():
        print("sample images dir does not exist")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(ka_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    for img in sample_imgs_dir.glob("*.png"):
        image = Image.open(img).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values,
                                           decoder_start_token_id=model.config.decoder_start_token_id,
                                           max_length=32
                                           )
            generated_text = tokenizer.decode(generated_ids[0].tolist())

        print(f"File: {img.name} -> Raw IDs: {generated_ids[0].tolist()}")  # Debug line
        print(f"File: {img.name} -> Recognized: {generated_text}")

