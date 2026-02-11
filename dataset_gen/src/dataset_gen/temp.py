import time

import cv2
import pandas as pd

from dataset_gen.utils import BASE_DIR
import albumentations as A
from pathlib import Path


def test_online_aug(img_path: Path):
    aug_pipeline = A.Compose([
        A.Rotate(
            limit=(-4, 4),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
            interpolation=cv2.INTER_LINEAR,
            p=0.5,
        ),
        A.GaussNoise(
            std_range=(0.1, 0.2),
            mean_range=(0.0, 0.0),
            per_channel=False,
            p=0.5,
        ),
        A.InvertImg(p=0.01),
        A.GaussianBlur(
            blur_limit=(7, 7),
            sigma_limit=(0.5, 0.5),
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=1.0,
            sigma=5.0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
            interpolation=cv2.INTER_LINEAR,
            p=0.5,
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
    images_dir = BASE_DIR / "data" / "raw"
    image_paths = list(images_dir.glob("**/*.png"))

    for image_path in image_paths:
        test_online_aug(image_path)

    print(f"augmented in {time.perf_counter() - t1:.2f} seconds")
