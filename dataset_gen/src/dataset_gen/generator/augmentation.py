import concurrent
import os
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
from augraphy import *
import albumentations as A

from dataset_gen.utils import BASE_DIR, PATHS


def augment_img(img_path: Path) -> None:
    """Augment a single image with Augraphy for realistic document like look."""
    ink_only_pipeline = AugraphyPipeline(
        ink_phase=[OneOf([InkBleed(), InkShifter(), LowInkRandomLines()], p=1)],
    )
    paper_only_pipeline = AugraphyPipeline(
        paper_phase=[OneOf([
            ColorPaper((10, 20), (10, 30)),
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
    images_dir = PATHS.synthetic_dir
    image_paths = list(images_dir.glob("**/*.png"))

    for image_path in image_paths:
        augment_img(image_path)

    print(f"augmented in {time.perf_counter() - t1:.2f} seconds")


def test_online_aug(img_path: Path):
    aug_pipeline = A.Compose([
        A.Rotate(
            limit=(-4, 4),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
            interpolation=cv2.INTER_LINEAR,
            p=0.7,
        ),
        A.GaussNoise(
            std_range=(0.1, 0.2),
            mean_range=(0.0, 0.0),
            per_channel=False,
            p=0.7,
        ),
        A.InvertImg(p=0.01),
        A.GaussianBlur(
            blur_limit=(7, 7),
            sigma_limit=(0.5, 0.5),
            p=0.7,
        ),
        A.ElasticTransform(
            alpha=1.0,
            sigma=5.0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
            interpolation=cv2.INTER_LINEAR,
            p=0.7,
        ),
    ])

    # Read image, apply pipeline and overwrite img
    img = cv2.imread(str(img_path))
    result = aug_pipeline(image=img)
    image_augmented = result["image"]
    cv2.imwrite(str(img_path), image_augmented)


def test_online_aug_images():
    print("\nAugmenting with albumentations...")
    t1 = time.perf_counter()
    images_dir = PATHS.synthetic_dir
    image_paths = list(images_dir.glob("**/*.png"))

    for image_path in image_paths:
        test_online_aug(image_path)

    print(f"augmented in {time.perf_counter() - t1:.2f} seconds")
