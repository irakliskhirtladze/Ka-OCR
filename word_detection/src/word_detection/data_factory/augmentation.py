import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from augraphy import *
import albumentations as A


def augment_doc(img: np.ndarray) -> np.ndarray:
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
        post_phase=[OneOf([
            DirtyRollers(),
            BadPhotoCopy(noise_type=2,
                         noise_iteration=(1, 1),  # Keep this at 1 to prevent "stacking" noise
                         noise_size=(1, 2),  # Smaller noise particles
                         noise_value=(0, 10),  # Very low intensity (Default is usually much higher)
                         )
        ], p=1)]
    )

    roll = random.random()

    if roll < 0.3:
        image_augmented = ink_only_pipeline(img)
    elif roll < 0.6:
        image_augmented = paper_only_pipeline(img)
    elif roll < 0.9:
        image_augmented = scanner_only_pipeline(img)
    else:  # 10% of the time, keep it perfectly clean
        image_augmented = img
    return image_augmented


def test_online_aug(img: np.ndarray) -> np.ndarray:
    aug_pipeline = A.Compose([
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

    result = aug_pipeline(image=img)
    return result["image"]
