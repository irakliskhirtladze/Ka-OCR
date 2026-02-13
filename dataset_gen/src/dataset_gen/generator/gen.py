import json
import re
import time

from trdg.generators import GeneratorFromStrings
from dataset_gen.utils import BASE_DIR
from pathlib import Path
import csv
import random
import os
from concurrent.futures import ProcessPoolExecutor


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
        georgian_only_pattern = re.compile(r'^[ა-ჰ]+$')
        valid_indices = [i for i, word in enumerate(word_list) if georgian_only_pattern.match(word)]
        if valid_indices:
            filtered_words = [word_list[i] for i in valid_indices]
            filtered_weights = [weights[i] for i in valid_indices]
            return random.choices(filtered_words, weights=filtered_weights, k=1)[0]

    return random.choices(word_list, weights=weights, k=1)[0]


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


def font_not_supports_specials(font_path: str) -> bool:
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
            if source_type < 0.95:
                text = get_random_word(word_list, weights, exclude_special_chars=True)
            else:
                text = get_random_sequence()
        else:
            if source_type < 0.92:
                text = get_random_word(word_list, weights)
            elif source_type < 0.98:
                text = get_random_sequence()
            else:
                text = get_random_number()

        strings.append(text)

    # Generate images one string at a time
    max_retries = 20
    for idx, text in enumerate(strings):
        img = None

        for attempt in range(max_retries):
            generator = GeneratorFromStrings(
                strings=[text],  # The text to be rendered (list of strings or a single string)
                fonts=[font_path],  # Paths to .ttf/.otf files; defaults to language-specific fonts if empty
                language="ka",  # Script/language code; helps TRDG pick appropriate fonts/dictionaries
                size=random.randint(40, 96),  # The height of the resulting image in pixels
                text_color="#202020,#505050",  # Hex code for font color (can also be a range like "#000,#fff")
                skewing_angle=2,  # text tilt degrees; used as a max value if random_skew is True
                random_skew=True,  # If True, randomly skews between 0 and skewing_angle
                background_type=1,  # 0: Gaussian Noise, 1: White, 2: Quid Paper, 3: Custom Image

                blur=0,  # Radius for Gaussian blur (0 = sharp, higher = blurrier)
                random_blur=False,  # If True, picks a random blur level between 0 and the blur value
                distorsion_type=0,  # 0: None, 1: Sine Wave, 2: Cosine Wave, 3: Random distortion
                distorsion_orientation=0,  # 0: Vertical, 1: Horizontal, 2: Both directions
                is_handwritten=False,  # If True, applies "handwriting" style offsets (often used with specific fonts)
                width=-1,  # Fixed width in pixels; -1 allows the width to scale with text length
                alignment=1,  # 0: Left, 1: Center, 2: Right (only effective if width is set)
                orientation=0,  # 0: Horizontal (standard), 1: Vertical text
                space_width=1.0,  # Multiplier for the width of the space character (e.g., 2.0 is double space)
                margins=(5, 5, 5, 5),  # Padding: (top, left, bottom, right) in pixels
                fit=False  # If True, crops the image to the tightest bounding box of the text
            )

            img = next(generator)

            if img is None:
                continue

            w, h = img.size
            if w >= 32 and h >= 32:
                break  # valid image
            img = None

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


def generate_imgs(num_images_per_font: int):
    """Generate synthetic images for all fonts.

    Args:
        num_images_per_font: Number of images to generate per font
        parallel_threshold: Use parallel processing if num_images_per_font >= this value
    """
    ka_font_dir = BASE_DIR / "src" / "dataset_gen" / "generator" / "fonts" / "ka"
    output_dir = BASE_DIR / "data" / "raw"
    dict_path = BASE_DIR / "src" / "dataset_gen" / "dictionaries" / "ka_dictionary.json"

    # Get all font files (ttf and otf)
    fonts = [str(f) for f in ka_font_dir.glob("*") if f.suffix.lower() in ['.ttf', '.otf']]
    if not fonts:
        print(f"No font files (.ttf, .otf) found in {ka_font_dir}")
        return

    # Check which fonts don't support special chars
    fonts_without_number_support = {}
    for font_path in fonts:
        no_specials = font_not_supports_specials(font_path)
        fonts_without_number_support[font_path] = no_specials

    fonts_no_specials = [Path(f).stem for f, no_support in fonts_without_number_support.items() if no_support]
    if fonts_no_specials:
        print(f"Fonts without number support: {', '.join(fonts_no_specials)}\n")

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

    # Run image generation either with multiple CPU cores, or sequentially
    t1 = time.perf_counter()
    num_workers = min(os.cpu_count() or 1, len(fonts))
    print(f"\nUsing parallel processing with {num_workers} workers...\n")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_generate_for_font, font_args))
    metadata = [item for result in results for item in result]
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
