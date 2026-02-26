import os
import time
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi
from dotenv import load_dotenv

from dataset_gen.generator.augmentation import augment_img, augment_images
from dataset_gen.generator.gen import generate_imgs
from dataset_gen.utils import BASE_DIR, Paths, PATHS


def write_version() -> None:
    """Writes version timestamp to file (UTC)."""
    current_timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    with open(PATHS.version_txt_path, "w", encoding="utf-8") as f:
        f.write(current_timestamp)
    print(f"Version {current_timestamp} (UTC) saved to {PATHS.version_txt_path}")


def extend_metadata_csv() -> None:
    """Take metadata.csv and extend it with real_augmented.csv"""
    synth_metadata_df = pd.read_csv(PATHS.synth_metadata_csv)
    real_aug_metadata_df = pd.read_csv(PATHS.real_metadata_path)

    df = pd.concat([synth_metadata_df, real_aug_metadata_df], ignore_index=True)
    df.to_csv(BASE_DIR / "data/metadata.csv", index=False)


def zip_dataset() -> None:
    """Zip the dataset preserving font subdirectory structure."""
    # Verify synth data exists
    if not PATHS.synthetic_dir.exists() or not PATHS.synthetic_metadata_path.exists():
        print("Error: Dataset not found. Run generate_imgs() first.")
        return

    # Find all images in synth data subdirectories
    synth_image_files = list(PATHS.synthetic_dir.glob("**/*.png"))
    if not synth_image_files:
        print("Error: No synthetic images found")
        return

    # Find all images in real data folder
    real_image_files = list(PATHS.real_images_dir.glob("**/*.png"))
    if not real_image_files:
        print("Error: No real images found")
        return

    # Create zip file preserving subdirectory structure
    print(f"\nCreating zip file with {len(synth_image_files)} images...")
    t1 = time.perf_counter()
    with zipfile.ZipFile(PATHS.zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # ============ add synthetic data folder ============
        zipf.write(PATHS.synthetic_metadata_path, arcname="synthetic/metadata.csv")

        for i, img_file in enumerate(synth_image_files):
            arcname = Path("synthetic") / img_file.relative_to(PATHS.synthetic_dir)
            zipf.write(img_file, arcname=str(arcname))
            if i % 100 == 0:  # Print every 100 to save console overhead
                print(f"\rAdding synthetic: {i + 1}/{len(synth_image_files)}...", end="", flush=True)

        # ============ add real images folder ============
        zipf.write(PATHS.real_metadata_path, arcname="real/metadata.csv")

        for i, img_file in enumerate(real_image_files):
            arcname = Path("real") / img_file.name  # Assuming real files aren't in subfolders
            zipf.write(img_file, arcname=str(arcname))
            print(f"\rAdding real: {i + 1}/{len(real_image_files)}...", end="", flush=True)

    zip_size_mb = PATHS.zip_path.stat().st_size / (1024 * 1024)
    t2 = time.perf_counter()
    print(f"\nCreated {PATHS.zip_path.name} ({zip_size_mb:.2f} MB)")
    print(f"Zipped in {(t2 - t1):.2f} seconds")

    # Generate version file
    write_version()


def dataset_to_hf() -> None:
    """Upload existing zip file to Hugging Face Hub."""
    load_dotenv()

    # Check for required environment variables
    hf_token = os.getenv("HF_TOKEN")
    hf_dataset_repo = os.getenv("HF_DATASET_REPO")

    if not hf_token:
        print("Error: HF_TOKEN not found in .env file")
        return

    if not hf_dataset_repo:
        print("Error: HF_DATASET_REPO not found in .env file")
        return

    zip_path = BASE_DIR / "data" / "ka-ocr.zip"

    if not zip_path.exists():
        print(f"Error: Zip file not found at {zip_path}")
        return

    version_path = BASE_DIR / "data" / "version.txt"
    if not version_path.exists():
        print(f"Warning: version.txt not found at {version_path}")

    # Push to Hugging Face
    print(f"\nPushing to Hugging Face: {hf_dataset_repo}")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo="ka-ocr.zip",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            token=hf_token
        )

        # Upload version.txt separately for easy version checking
        if version_path.exists():
            api.upload_file(
                path_or_fileobj=str(version_path),
                path_in_repo="version.txt",
                repo_id=hf_dataset_repo,
                repo_type="dataset",
                token=hf_token
            )

        print(f"Successfully uploaded to https://huggingface.co/datasets/{hf_dataset_repo}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")


def main() -> None:
    # setup necessary paths
    PATHS.synthetic_dir.mkdir(parents=True, exist_ok=True)

    # Image generation
    while True:
        user_input = input("Do you want to generate synthetic images? (Y/N): ")
        if user_input.lower() == "y":
            while True:
                user_input = input("\nHow many images per font would you like to generate?: ")
                if not user_input.isdigit() or int(user_input) <= 0:
                    print("Please enter a positive integer.")
                    continue
                else:
                    break
            generate_imgs(int(user_input))
            break
        elif user_input.lower() == "n":
            break
        else:
            print("Please enter a valid input.")

    # Image augmentation
    augment_images()

    # # Test online augmentation with albumentations
    # from dataset_gen.temp import test_online_aug_images
    # test_online_aug_images()

    # Zipping the dataset
    while True:
        user_input = input("\nDo you want to zip the dataset? (Y/N): ")
        if user_input.lower() == "y":
            zip_dataset()
            break
        elif user_input.lower() == "n":
            print("Zipping cancelled.")
            break
        else:
            print("Please enter either 'Y' or 'N'.")

    # Uploading dataset to HF
    while True:
        user_input = input("\nDo you want to upload zipped dataset ot Hugging Face? (Y/N) ?: ")
        if user_input.lower() == "y":
            dataset_to_hf()
            break
        elif user_input.lower() == "n":
            print("HF upload cancelled.")
            break
        else:
            print("Please enter either 'Y' or 'N'.")


if __name__ == "__main__":
    main()
