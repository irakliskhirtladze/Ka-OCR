import os
import shutil

from torchvision.utils import save_image
from PIL import Image
from ml_training.setup import Paths, check_env
import torch
from pathlib import Path
from transformers import TrOCRProcessor
from huggingface_hub import HfApi
from dotenv import load_dotenv


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
    ka_model_path = paths.output_dir / "best_model.pt"  # The path to .pt file

    if check_env() == "colab":
        sample_imgs_dir = Path("/content/drive/MyDrive/Colab Notebooks/trocr-ka/data/sample_images")
    else:
        sample_imgs_dir = paths.base_dir / "sample_images"

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

        print(f"File: {img.name} -> Recognized: {generated_text}")


def save_final_model(paths: Paths, model: torch.nn.Module, processor: TrOCRProcessor):
    """Save the final trained model and processor."""
    best_weights_path = paths.output_dir / "best_model.pt"
    export_dir = paths.output_dir / "georgian_trocr_deployment"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save model and processor
    if best_weights_path.exists():
        print(f"Loading best weights from {best_weights_path} before export...")
        model.load_state_dict(torch.load(best_weights_path, map_location='cpu'))

    model.save_pretrained(export_dir, safe_serialization=True)
    processor.save_pretrained(export_dir)

    print(f"Model and processor saved to: {export_dir}")

    # Sync to Drive on Colab
    if check_env() == "colab":
        # Create a zip of the output for easy download
        zip_path = paths.output_dir / "final_model.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', export_dir)

        # Copy zip to Drive
        shutil.copy(zip_path, paths.drive_output_dir / "final_model.zip")
        print(f"Model zip synced to Drive: {paths.drive_output_dir / 'final_model.zip'}")


def upload_to_hf(paths: Paths) -> None:
    """Upload trained model to Hugging Face Hub."""
    load_dotenv(paths.base_dir / ".env")
    hf_repo = os.getenv("HF_MODEL_REPO")
    repo_id = f"irskhirtladze/{hf_repo}"

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=paths.output_dir / "georgian_trocr_deployment",
        repo_id=repo_id,
        repo_type="model"
    )

    print(f"Model uploaded to: https://huggingface.co/{repo_id}")
