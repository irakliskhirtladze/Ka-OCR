from pathlib import Path
from PIL import Image


def check_img_quality(raw_path: Path):
    print(f"looking in dir: {raw_path}")
    images = list(raw_path.glob("**/*.png"))
    print(len(images))
    for image in images:
        img = Image.open(image)
        if img is None:
            print(f"invalid image: {image}")
            continue

        if img.width < 30 or img.height < 30:
            print(f"invalid image: {image} because of too small")


if __name__ == "__main__":
    check_img_quality(Path(__file__).resolve().parent.parent.parent.parent / "dataset_gen/data/raw")
