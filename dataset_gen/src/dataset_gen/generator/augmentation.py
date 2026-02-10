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
    cv2.setNumThreads(0)
    ink_phase = [
        InkBleed(
            intensity_range=(0.4, 0.7),
            kernel_size=random.choice([(5, 5), (3, 3)]),
            severity=(0.2, 0.4),
            p=0.8,
        ),
        OneOf(
            [
                InkShifter(
                    text_shift_scale_range=(18, 27),
                    text_shift_factor_range=(1, 4),
                    text_fade_range=(0, 2),
                    blur_kernel_size=(5, 5),
                    blur_sigma=0,
                    noise_type="random",
                ),
                BleedThrough(
                    intensity_range=(0.1, 0.3),
                    color_range=(32, 224),
                    ksize=(17, 17),
                    sigmaX=1,
                    alpha=random.uniform(0.1, 0.2),
                    offsets=(10, 20),
                ),
            ],
            p=0.8,
        ),
    ]

    paper_phase = [
        ColorPaper(
            hue_range=(0, 255),
            saturation_range=(10, 40),
            p=0.8,
        ),
        OneOf(
            [
                DelaunayTessellation(
                    n_points_range=(500, 800),
                    n_horizontal_points_range=(500, 800),
                    n_vertical_points_range=(500, 800),
                    noise_type="random",
                    color_list="default",
                    color_list_alternate="default",
                ),
                PatternGenerator(
                    imgx=random.randint(256, 512),
                    imgy=random.randint(256, 512),
                    n_rotation_range=(10, 15),
                    color="random",
                    alpha_range=(0.25, 0.5),
                ),
                VoronoiTessellation(
                    mult_range=(50, 80),
                    seed=19829813472,
                    num_cells_range=(500, 1000),
                    noise_type="random",
                    background_value=(200, 255),
                ),
            ],
            p=0.8,
        ),
        AugmentationSequence(
            [
                NoiseTexturize(
                    sigma_range=(3, 10),
                    turbulence_range=(2, 5),
                    p=0.5,
                ),
                BrightnessTexturize(
                    texturize_range=(0.9, 0.99),
                    deviation=0.03,
                    p=0.5,
                ),
            ],
        ),
    ]

    post_phase = [
        OneOf(
            [
                DirtyDrum(
                    line_width_range=(1, 6),
                    line_concentration=random.uniform(0.05, 0.15),
                    direction=random.randint(0, 2),
                    noise_intensity=random.uniform(0.6, 0.95),
                    noise_value=(64, 224),
                    ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
                    sigmaX=0,
                ),
                DirtyRollers(
                    line_width_range=(2, 32),
                    scanline_type=0,
                ),
            ],
            p=0.8,
        ),
        SubtleNoise(
            subtle_range=random.randint(5, 10),
            p=0.5,
        ),
        Jpeg(
            quality_range=(25, 95),
            p=0.5,
        ),
    ]

    # Set up pipeline
    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase
    )

    # Read image, apply pipeline and overwrite img
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping unreadable image: {img_path}")
        return
    image_augmented = pipeline(img)

    tmp_path = img_path.with_name(f"{img_path.stem}.tmp{os.getpid()}{img_path.suffix}")
    try:
        ok = cv2.imwrite(str(tmp_path), image_augmented)
        if not ok:
            print(f"Failed to write augmented image: {img_path}")
            return
        os.replace(tmp_path, img_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def augment_images() -> None:
    print("\nAugmenting...")
    t1 = time.perf_counter()
    images_dir = BASE_DIR / "data" / "raw"
    image_paths = list(images_dir.glob("**/*.png"))
    num_workers = min(os.cpu_count() or 1, len(image_paths))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(augment_img, image_paths)
    print(f"augmented in {time.perf_counter() - t1:.2f} seconds")
