import concurrent
import os
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
from augraphy import *

from dataset_gen.utils import BASE_DIR


def augment_img(img_path: Path) -> None:
    """Augment a single image with Augraphy for realistic document like look."""
    ink_only_pipeline = AugraphyPipeline(
        ink_phase=[OneOf([InkBleed(), InkShifter(), LowInkRandomLines()], p=1)],
    )
    paper_only_pipeline = AugraphyPipeline(
        paper_phase=[OneOf([
            ColorPaper((0, 10), (0, 20)),
            # PaperFactory(generate_texture_background_type="normal")
        ],
            p=1)],
    )
    scanner_only_pipeline = AugraphyPipeline(
        post_phase=[OneOf([DirtyRollers(), BadPhotoCopy(noise_type=2)], p=1)]
    )

    img = cv2.imread(str(img_path))
    roll = random.random()

    if roll < 0.3:
        image_augmented = ink_only_pipeline(img)
    elif roll < 0.6:
        image_augmented = paper_only_pipeline(img)
    elif roll < 0.9:
        image_augmented = scanner_only_pipeline(img)
    else:  # 10% of the time, keep it perfectly clean
        image_augmented = img
    cv2.imwrite(str(img_path), image_augmented)


def augment_images() -> None:
    print("\nAugmenting...")
    t1 = time.perf_counter()
    images_dir = BASE_DIR / "data" / "raw"
    image_paths = list(images_dir.glob("**/*.png"))

    # num_workers = min(os.cpu_count() or 1, len(image_paths))
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     executor.map(augment_img, image_paths)

    for image_path in image_paths:
        augment_img(image_path)

    print(f"augmented in {time.perf_counter() - t1:.2f} seconds")
