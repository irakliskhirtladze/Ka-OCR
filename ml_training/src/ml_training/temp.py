import os
import torch
from torchvision.utils import save_image


def save_debug_samples(dataloader, tokenizer, output_dir, n_images=16):
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