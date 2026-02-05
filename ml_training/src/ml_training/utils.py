import os
from pathlib import Path

import torch
from torchvision.utils import save_image
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ml_training.dataset import GeorgianTokenizer
from ml_training.setup import Paths


def save_debug_samples(dataloader, tokenizer, output_dir, n_images=50):
    """Saves un-normalized augmented samples to disk for visual inspection."""
    debug_path = os.path.join(output_dir, "debug_augs")
    os.makedirs(debug_path, exist_ok=True)

    # Get one batch
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"]  # Shape: [batch_size, 3, 384, 384]
    labels = batch["labels"]

    # Take only the number of images requested
    pixel_values = pixel_values[:n_images]

    for i in range(pixel_values.shape[0]):
        # 1. Un-normalize: Bring from (~ -1 to 1) back to (0 to 1)
        # We do this by finding the min/max of the specific image
        img_tensor = pixel_values[i]
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        img_tensor = (img_tensor - img_min) / (img_max - img_min)

        # 2. Get the label text for the filename (sanitize for Windows/Linux paths)
        clean_label = labels[i][labels[i] != -100]
        decoded_text = tokenizer.decode(clean_label.tolist())
        safe_text = "".join([c for c in decoded_text if c.isalnum() or c in (' ', '_')]).strip()[:20]

        # 3. Save to disk
        fname = f"sample_{i}_{safe_text}.png"
        save_image(img_tensor, os.path.join(debug_path, fname))

    print(f"--- Saved {n_images} debug images to: {debug_path} ---")


def test_against_real_images(paths: Paths, model, processor):
    print("Testing against real images...")
    ka_model_path = paths.drive_output_dir / "best_model.pt"  # The path to your .pt file
    sample_imgs_dir = Path("content/drive/MyDrive/Colab Notebooks/trocr-ka/data/")
    if not sample_imgs_dir.exists():
        print("sample images dir does not exist")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(ka_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for img in sample_imgs_dir.glob("*.png"):
        print(img)
        image = Image.open(img).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"File: {img.name} -> Recognized: {generated_text}")

