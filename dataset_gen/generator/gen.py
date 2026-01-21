import json
import time
from datetime import datetime

from trdg.generators import GeneratorFromStrings
from utils import BASE_DIR
from pathlib import Path
import csv
import random
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from huggingface_hub import HfApi
from dotenv import load_dotenv


def load_dictionary(dict_path: Path = None) -> tuple[list, list]:
    """Load dictionary once and return words with weights for efficient sampling"""
    if dict_path is None:
        dict_path = BASE_DIR / "generator" / "dictionaries" / "ka_dictionary.json"

    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data["words"]
    word_list = [w["word"] for w in words]
    weights = [w["weight"] for w in words]
    return word_list, weights


def get_random_word(word_list: list, weights: list, exclude_special_chars: bool = False) -> str:
    """Get a random word from the dictionary with frequency weighting
    
    Args:
        word_list: List of words
        weights: Corresponding weights
        exclude_special_chars: If True, exclude words with hyphens or numbers
    """
    if exclude_special_chars:
        # Filter out words containing hyphens or numbers
        valid_indices = [i for i, word in enumerate(word_list) 
                        if '-' not in word and not any(c.isdigit() for c in word)]
        if valid_indices:
            filtered_words = [word_list[i] for i in valid_indices]
            filtered_weights = [weights[i] for i in valid_indices]
            chosen = random.choices(filtered_words, weights=filtered_weights, k=1)[0]
            return chosen
    
    chosen = random.choices(word_list, weights=weights, k=1)[0]
    return chosen


def get_random_sequence(length: int = None) -> str:
    """Generate random sequence of Georgian characters"""
    chars = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
    if length is None:
        length = random.randint(3, 12)
    return "".join(random.choice(chars) for _ in range(length))


def get_random_number() -> str:
    """Generate random number or date"""
    choice = random.random()

    if choice < 0.3:  # Simple number
        return str(random.randint(0, 9999))
    elif choice < 0.6:  # Date DD.MM.YYYY
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(1900, 2025)
        return f"{day:02d}.{month:02d}.{year}"
    elif choice < 0.8:  # Year only
        return str(random.randint(1800, 2025))
    else:  # Phone-like number
        return f"+995{random.randint(500000000, 599999999)}"


def font_not_supports_numbers(font_path: str) -> bool:
    """Check if font does NOT support Latin numerals (0-9)"""
    font_name = Path(font_path).stem.lower()
    # Known Georgian-only fonts without number support
    fonts_without_numbers = ['3d_unicode', 'notosansgeorgian', 'alkroundedmtav-medium', 'alkroundednusx-medium']
    return font_name in fonts_without_numbers


def _generate_for_font(args: tuple) -> list[dict]:
    """Worker function: generates all images for a single font.
    
    Args:
        args: Tuple of (font_path, num_images, word_list, weights, output_dir, no_number_support)
    
    Returns:
        List of metadata dicts for generated images
    """
    font_path, num_images, word_list, weights, output_dir, no_number_support = args
    font_name = Path(font_path).stem
    metadata = []
    
    # Generate text strings for this font
    strings = []
    for _ in range(num_images):
        source_type = random.random()
        
        if no_number_support:
            if source_type < 0.9:
                text = get_random_word(word_list, weights, exclude_special_chars=True)
            else:
                text = get_random_sequence()
        else:
            if source_type < 0.9:
                text = get_random_word(word_list, weights)
            elif source_type < 0.97:
                text = get_random_sequence()
            else:
                text = get_random_number()
        
        strings.append(text)
    
    # Generate images one string at a time
    for idx, text in enumerate(strings):
        generator = GeneratorFromStrings(
            strings=[text],
            fonts=[font_path],
            language="ka",
            size=random.randint(32, 96),
            skewing_angle=random.randint(0, 10),
            random_skew=True,
            blur=random.randint(0, 1),
            random_blur=True,
            distorsion_type=random.randint(0, 3),  # 0=none, 1=sine, 2=cosine, 3=random
            distorsion_orientation=random.randint(0, 2),
            background_type=random.randint(0, 2),  # 0=gaussian, 1=plain white, 2=quasicrystal, 3=image
            text_color="#000000,#1a1a1a,#333333,#2b1a1a,#1a0f0f,#3d2b2b,#4a0000,#2d1f1f",
            margins=(random.randint(0, 10), random.randint(0, 10), random.randint(0, 10),
                     random.randint(0, 10)),
            fit=random.choice([True, False]),
        )

        img = next(generator)

        if img is None:
            continue

        image_group_dir = Path(output_dir) / font_name
        image_group_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{font_name}_{idx:04d}.png"
        img_save_path = Path(image_group_dir) / file_name

        img.save(img_save_path)
        metadata.append({"file_name": f"{image_group_dir.stem}/{file_name}", "text": text})

    print(f"\nGenerated {len(metadata)} images for {font_name}")

    return metadata


def write_version(file_path: Path):
    """Writes version timestamp to file (UTC)."""
    current_timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(current_timestamp)
    print(f"Version {current_timestamp} (UTC) saved to {file_path}")


def generate_imgs(num_images_per_font: int):
    """Generate synthetic images for all fonts.

    Args:
        num_images_per_font: Number of images to generate per font
        parallel_threshold: Use parallel processing if num_images_per_font >= this value
    """
    ka_font_dir = BASE_DIR / "generator" / "fonts" / "ka"
    output_dir = BASE_DIR / "data" / "raw"
    dict_path = BASE_DIR / "generator" / "dictionaries" / "ka_dictionary.json"

    # Get all font files (ttf and otf)
    fonts = [str(f) for f in ka_font_dir.glob("*") if f.suffix.lower() in ['.ttf', '.otf']]
    if not fonts:
        print(f"No font files (.ttf, .otf) found in {ka_font_dir}")
        return

    # Check which fonts don't support numbers
    fonts_without_number_support = {}
    for font_path in fonts:
        no_numbers = font_not_supports_numbers(font_path)
        fonts_without_number_support[font_path] = no_numbers

    fonts_no_nums = [Path(f).stem for f, no_support in fonts_without_number_support.items() if no_support]
    if fonts_no_nums:
        print(f"Fonts without number support: {', '.join(fonts_no_nums)}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating images for {len(fonts)} fonts...")
    print(f"Images per font: {num_images_per_font}")
    print(f"Total images to generate: {len(fonts) * num_images_per_font}")
    print(f"Distribution: 90% real words, 7% random sequences, 3% numbers (font-dependent)")

    # Load dictionary
    print("\nLoading dictionary...")
    word_list, weights = load_dictionary(dict_path)
    print(f"Loaded {len(word_list)} words with frequency weights")

    # Prepare args for each font
    font_args = [
        (font_path, num_images_per_font, word_list, weights, str(output_dir), fonts_without_number_support[font_path])
        for font_path in fonts
    ]

    # Ask user model of image generation
    while True:
        user_input = input("\nDo you want to do parallel image generation? (Y/N): ")
        if user_input.lower() == "y":
            use_parallel = True
            break
        elif user_input.lower() == "n":
            use_parallel = False
            break
        else:
            print("Please enter either 'Y' or 'N'.")

    # Run image generation either with multiple CPU cores, or sequentially
    t1 = time.perf_counter()
    if use_parallel:
        num_workers = min(os.cpu_count() or 1, len(fonts))
        print(f"\nUsing parallel processing with {num_workers} workers...\n")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_generate_for_font, font_args))

        metadata = [item for result in results for item in result]
    else:
        print("\nUsing sequential processing...\n")
        metadata = []
        for font_idx, args in enumerate(font_args):
            font_name = Path(args[0]).stem
            print(f"[{font_idx+1}/{len(fonts)}] Processing font: {font_name}")
            result = _generate_for_font(args)
            metadata.extend(result)
    t2 = time.perf_counter()
    print(f"\nDone in {(t2 - t1)} seconds")

    # Write Labels to CSV
    csv_path = BASE_DIR / "data" / "metadata.csv"
    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\n✓ Finished! {len(metadata)} images saved to {output_dir}")
    print(f"✓ Labels saved to {csv_path}")

    # Write version file
    write_version(BASE_DIR / "data" / "version.txt")


def zip_dataset():
    """Zip the dataset preserving font subdirectory structure."""
    data_dir = BASE_DIR / "data"
    raw_dir = data_dir / "raw"
    metadata_file = data_dir / "metadata.csv"
    version_file = data_dir / "version.txt"
    zip_path = data_dir / "ka-ocr.zip"

    # Verify data exists
    if not raw_dir.exists() or not metadata_file.exists():
        print("Error: Dataset not found. Run generate_imgs() first.")
        return

    # Find all images in subdirectories
    image_files = list(raw_dir.glob("**/*.png"))
    if not image_files:
        print("Error: No images found in data/raw/")
        return

    print(f"\nCreating zip file with {len(image_files)} images...")

    # Create zip file preserving subdirectory structure
    t1 = time.perf_counter()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        num_images = len(image_files)
        for i, img_file in enumerate(image_files):
            print(f"\radding image {i+1}/{num_images}...", end="", flush=True)
            arcname = img_file.relative_to(raw_dir)
            zipf.write(img_file, arcname=arcname)

        # Add metadata.csv and version.txt to zip root
        zipf.write(metadata_file, arcname="metadata.csv")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    t2 = time.perf_counter()
    print(f"\nCreated {zip_path.name} ({zip_size_mb:.2f} MB)")
    print(f"Zipped in {(t2 - t1):.2f} seconds")


def dataset_to_hf():
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

