import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
from urllib.parse import unquote

from augraphy.base.augmentationpipeline import AugraphyPipeline
from augraphy.augmentations.inkbleed import InkBleed
from augraphy.augmentations.colorpaper import ColorPaper
from augraphy.augmentations.brightness import Brightness

from dataset_gen.utils import PATHS


def file_renamer() -> None:
    files = Path("data/real").glob("*.png")
    for i, file in enumerate(files, start=1):
        new_path = file.with_stem(str(i).zfill(4))
        file.rename(new_path)


def clean_filename(path_str: str) -> str:
    # 1. Decode %5C to \
    decoded_path = unquote(path_str)
    return Path(decoded_path).name


def labels_json_to_df() -> pd.DataFrame:
    with open(PATHS.real_json_path, "r", encoding="utf8") as f:
        data: list = json.load(f)

    df = pd.DataFrame(data)
    df = df[["image", "manual_label"]]
    df.rename(columns={"image": "file_name", "manual_label": "text"}, inplace=True)
    df["file_name"] = df["file_name"].apply(clean_filename)

    # check actual file names keep in df only entries of existing files
    imgs = {f.name for f in PATHS.real_images_dir.glob("*.png")}
    df = df[df["file_name"].isin(imgs)].copy()
    return df


def save_real_metadata(df: pd.DataFrame) -> None:
    df.to_csv(PATHS.real_metadata_path, index=False)


# def augment_real_data(real_imgs_dir: Path, labels_df: pd.DataFrame, num_copies: int, output_dir: Path) -> None:
#     """Takes real images, creates augmented copies of them, creates corresponding labels in df and saves files"""
#     real_imgs = real_imgs_dir.glob("*.png")
#
#     ink_phase = [InkBleed(intensity_range=(0.4, 0.7), kernel_size=(3, 3), p=1)]
#     paper_phase = [ColorPaper(hue_range=(0, 255), saturation_range=(0, 30), p=0.8)]
#     post_phase = [Brightness(brightness_range=(0.8, 1.1), p=0.8)]
#     pipeline = AugraphyPipeline(
#         ink_phase=ink_phase,
#         paper_phase=paper_phase,
#         post_phase=post_phase
#     )
#
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     augmented_labels_list = []
#     for file_path in real_imgs:
#         print(f"processing {file_path.name}")
#         label_row = labels_df[labels_df["file_name"] == file_path.name]
#         label_text = label_row.iloc[0]["text"]
#         image = cv2.imread(str(file_path))
#
#         # Check if image is too small for Augraphy (height < 30 or width < 30)
#         h, w = image.shape[:2]
#         if h < 30 or w < 30:
#             # Resize so the smallest dimension is at least 30
#             new_h = max(30, h)
#             new_w = max(30, w)
#             image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#
#         for i in range(num_copies):
#             aug_image = pipeline.augment(image)['output']
#             new_filename = f"{file_path.stem}_aug_{i}.png"
#             cv2.imwrite(str(output_dir / new_filename), aug_image)
#
#             # create dict of file name and label and add to list
#             augmented_labels_list.append({"file_name": new_filename, "text": label_text})
#
#     augmented_labels_df = pd.DataFrame(augmented_labels_list)
#     final_df = pd.concat([labels_df, augmented_labels_df], ignore_index=True)
#     print(f"Originals: {len(labels_df)}")
#     print(f"Augmented: {len(augmented_labels_df)}")
#     print(f"Total:     {len(final_df)}")
#     final_df["file_name"] = final_df["file_name"].apply(lambda x: f"real_augmented/{x}")
#     final_df.to_csv(BASE_DIR / "data" / "real_augmented.csv", index=False)
#
#
# def copy_reals_to_aug() -> None:
#     """Copy and paste real images to augmented images dir"""
#     real_imgs_dir = BASE_DIR / "data" / "real"
#     augmented_imgs_dir = BASE_DIR / "data" / "real_augmented"
#
#     real_imgs = real_imgs_dir.glob("*.png")
#     for img_path in real_imgs:
#         shutil.copy2(img_path, augmented_imgs_dir / img_path.name)
#
#     print(f"Successfully copied images to {augmented_imgs_dir}")


if __name__ == "__main__":
    real_labels_df = labels_json_to_df()
    save_real_metadata(real_labels_df)
