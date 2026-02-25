import itertools
import os
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from random import randint, choice
from huggingface_hub import HfApi
from dotenv import load_dotenv

from word_detection.data_factory.augmentation import augment_doc
from word_detection.utils import PATHS


class DocumentGenerator:
    """Renders text on a single image and saves bounding boxes of each word"""

    def __init__(self, img_name: str, width: int, height: int, font_path: Path, font_size: int, text: str) -> None:
        self.img_name = img_name
        self.width = width
        self.height = height
        self.text = text
        self.font_path = font_path
        self.font_size = font_size
        self.font = ImageFont.truetype(str(font_path), self.font_size)

        if self.width < 640 or self.height < 640:
            raise Exception("width and height must be greater than 640")

    def _get_yolo_coords(self, bbox: tuple[int, int, int, int]) -> tuple[int, float, float, float, float]:
        """[left, top, right, bottom] YOLO"""
        l, t, r, b = bbox
        w = (r - l) / self.width
        h = (b - t) / self.height
        x_c = (l + r) / (2 * self.width)
        y_c = (t + b) / (2 * self.height)
        return 0, x_c, y_c, w, h  # 0 for class "word"

    def render_document(self, mode: str = "block") -> tuple[np.ndarray, list[tuple]]:
        """Main method to render document-like image"""
        cv_img = self._generate_background()

        if mode == "table":
            return self._render_table(cv_img)
        else:
            return self._render_block(cv_img)

    def _render_block(self, cv_img: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        """Generate and render block of text, return tuple of image and bbox list on it"""
        pil_img = Image.fromarray(cv_img)
        draw = ImageDraw.Draw(pil_img)

        words = self._get_text_slice()
        space_w = draw.textbbox((0, 0), " ", font=self.font)[2]
        line_height = self.font_size + 10

        x_cursor, y_cursor = 40, 40  # initial pos

        bbox_list = []
        for word in words:
            word = word.strip()
            if not word:
                continue

            try:
                bbox = draw.textbbox((x_cursor, y_cursor), word, font=self.font)
            except OSError:
                print(f"Skipping corrupt word/font combo: {word} with {self.font_path.name}")
                continue
            word_w = bbox[2] - bbox[0]

            # if word does not fit on line go to next line
            if x_cursor + word_w > self.width - 40:
                x_cursor = 40
                y_cursor += line_height

            # stop if exceeding image height
            if y_cursor + line_height > self.height - 40:
                break

            x1, y1, x2, y2 = draw.textbbox((x_cursor, y_cursor), word, font=self.font)
            draw.text((x_cursor, y_cursor), word, font=self.font, fill=(0, 0, 0))
            yolo_bbox = (
                0,  # class label
                (x1 + x2) / 2 / self.width,  # center_x normalized
                (y1 + y2) / 2 / self.height,  # center_y normalized
                (x2 - x1) / self.width,  # width normalized
                (y2 - y1) / self.height,  # height normalized
            )

            # move cursor
            x_cursor += word_w + space_w
            bbox_list.append(yolo_bbox)

        return np.array(pil_img), bbox_list

    def _render_table(self, cv_img: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        """Generate and render tabular doc with texts, return image and bbox list"""
        rows, cols = randint(5, 12), randint(2, 5)
        row_h = self.height // (rows + 2)
        col_w = self.width // cols

        PADDING = 6
        ALIGNMENTS = ["left", "center", "right"]
        VALIGNMENTS = ["top", "middle", "bottom"]

        for i in range(rows + 1):
            cv2.line(cv_img, (0, (i + 1) * row_h), (self.width, (i + 1) * row_h), (200, 200, 200), randint(1, 3))
        for j in range(cols + 1):
            cv2.line(cv_img, (j * col_w, row_h), (j * col_w, (rows + 1) * row_h), (200, 200, 200), randint(1, 3))

        pil_img = Image.fromarray(cv_img)
        draw = ImageDraw.Draw(pil_img)
        words = self._get_text_slice()
        space_w = draw.textbbox((0, 0), " ", font=self.font)[2]

        def wrap_words_to_cell(words: list[str], start_idx: int, max_w: int, max_h: int) -> tuple[list[str], int]:
            lines = []
            current_line_words = []
            idx = start_idx

            while idx < len(words):
                candidate = " ".join(current_line_words + [words[idx]])
                try:
                    bbox = draw.textbbox((0, 0), candidate, font=self.font)
                except OSError:
                    print(f"Skipping corrupt word/font combo: '{words[idx]}' with {self.font_path.name}")
                    idx += 1
                    continue
                line_w = bbox[2] - bbox[0]
                line_h = bbox[3] - bbox[1]

                if line_w <= max_w:
                    current_line_words.append(words[idx])
                    idx += 1
                else:
                    if not current_line_words:
                        current_line_words.append(words[idx])
                        idx += 1
                    lines.append(" ".join(current_line_words))
                    current_line_words = []

                projected_h = (len(lines) + 1) * line_h
                if projected_h > max_h and current_line_words:
                    break

            if current_line_words:
                lines.append(" ".join(current_line_words))

            while lines:
                total_h = sum(
                    draw.textbbox((0, 0), line, font=self.font)[3] - draw.textbbox((0, 0), line, font=self.font)[1]
                    for line in lines
                )
                if total_h <= max_h:
                    break
                lines.pop()

            placed_words = sum(len(line.split()) for line in lines)
            return lines, start_idx + placed_words

        word_idx = 0
        bbox_list = []

        for r in range(rows):
            h_align = choice(ALIGNMENTS)
            v_align = choice(VALIGNMENTS)

            for c in range(cols):
                if word_idx >= len(words):
                    break

                cell_left = c * col_w + PADDING
                cell_top = (r + 1) * row_h + PADDING
                cell_right = (c + 1) * col_w - PADDING
                cell_bottom = (r + 2) * row_h - PADDING
                cell_w = cell_right - cell_left
                cell_h = cell_bottom - cell_top

                lines, word_idx = wrap_words_to_cell(words, word_idx, cell_w, cell_h)
                if not lines:
                    continue

                line_sizes = [draw.textbbox((0, 0), line, font=self.font) for line in lines]
                line_heights = [b[3] - b[1] for b in line_sizes]
                line_widths = [b[2] - b[0] for b in line_sizes]
                total_block_h = sum(line_heights)
                line_gap = 2

                if v_align == "top":
                    start_y = cell_top
                elif v_align == "middle":
                    start_y = cell_top + (cell_h - total_block_h) // 2
                else:
                    start_y = cell_bottom - total_block_h

                y = start_y
                for line, lw, lh in zip(lines, line_widths, line_heights):
                    if h_align == "left":
                        x = cell_left
                    elif h_align == "center":
                        x = cell_left + (cell_w - lw) // 2
                    else:
                        x = cell_right - lw

                    x_cursor = x
                    for word in line.split():
                        word = word.strip()
                        if not word:
                            continue

                        try:
                            x1, y1, x2, y2 = draw.textbbox((x_cursor, y), word, font=self.font)
                        except OSError:
                            print(f"Skipping corrupt word/font combo: {word} with {self.font_path.name}")
                            continue

                        draw.text((x_cursor, y), word, font=self.font, fill=(0, 0, 0))
                        yolo_bbox = (
                            0,  # class label
                            (x1 + x2) / 2 / self.width,  # center_x normalized
                            (y1 + y2) / 2 / self.height,  # center_y normalized
                            (x2 - x1) / self.width,  # width normalized
                            (y2 - y1) / self.height,  # height normalized
                        )
                        bbox_list.append(yolo_bbox)
                        x_cursor += (x2 - x1) + space_w

                    y += lh + line_gap

        return np.array(pil_img), bbox_list

    def _font_not_supports_specials(self) -> bool:
        """Check if font does NOT support special characters"""
        fonts_without_numbers = ['3d_unicode', 'notosansgeorgian', 'alkroundedmtav-medium', 'alkroundednusx-medium']
        return self.font_path.stem.lower() in fonts_without_numbers

    def _generate_background(self) -> np.ndarray:
        image = np.ones((self.height, self.width, 3), np.uint8) * 255
        return image

    def _get_text_slice(self) -> list[str]:
        """returns list of words. filter out words with special characters if font not supports them"""
        words = self.text.split()

        if self._font_not_supports_specials():
            ka_chars = set("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
            filtered_words = [word for word in words if word and all(char in ka_chars for char in word)]
        else:
            filtered_words = [word for word in words if word]

        total_count = len(filtered_words)

        max_to_take = min(400, total_count)
        num_to_take = randint(10, max_to_take)

        start_idx = randint(0, total_count - num_to_take)
        return filtered_words[start_idx: start_idx + num_to_take]


def _generate_for_font(font_path: Path, target_dir: Path, texts: list[str]) -> None:
    """Generate 29 document images for a single font and save to target_dir."""
    for i in range(29):
        file_name = f"{font_path.stem}-{i}"
        doc_generator = DocumentGenerator(
            file_name,
            randint(640, 2000),
            randint(640, 2000),
            font_path,
            randint(11, 28),
            choice(texts)
        )

        img, bboxes = doc_generator.render_document(mode=choice(["table", "block"]))
        img = augment_doc(img)

        cv2.imwrite(str(target_dir / file_name) + ".png", img)
        with open(str(target_dir / file_name) + ".txt", "w", encoding="utf-8") as f:
            for bbox in bboxes:
                cls, cx, cy, w, h = bbox
                f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def generate_docs() -> None:
    texts: list[str] = []
    for text_file_path in PATHS.text_files_dir.glob("*.txt"):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            texts.append(text_file.read())

    font_files = sorted(
        itertools.chain(PATHS.fonts_dir.glob("*.ttf"), PATHS.fonts_dir.glob("*.otf"))
    )
    split_idx = int(len(font_files) * 0.9)
    train_fonts = font_files[:split_idx]
    val_fonts = font_files[split_idx:]

    t1 = time.perf_counter()
    for font_path in train_fonts:
        _generate_for_font(font_path, PATHS.train_dir, texts)
    for font_path in val_fonts:
        _generate_for_font(font_path, PATHS.val_dir, texts)

    print(f"Finished in {time.perf_counter() - t1:.1f}s — "
          f"{len(train_fonts)} train fonts, {len(val_fonts)} val fonts.")


def zip_dataset() -> None:
    dataset_files = list(PATHS.dataset_dir.glob("*"))
    print(f"\nCreating zip file with {len(dataset_files)} files...")
    t1 = time.perf_counter()
    with zipfile.ZipFile(PATHS.zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, file in enumerate(dataset_files):
            arcname = Path("YOLO_ka_words") / file.name  # if files aren't in subfolders
            zipf.write(file, arcname=str(arcname))

    zip_size_mb = PATHS.zip_path.stat().st_size / (1024 * 1024)
    t2 = time.perf_counter()
    print(f"\nCreated {PATHS.zip_path.name} ({zip_size_mb:.2f} MB)")
    print(f"Zipped in {(t2 - t1):.2f} seconds\n")


def dataset_to_hf() -> None:
    """Upload existing zip file to Hugging Face Hub."""
    load_dotenv(str(PATHS.env_path))
    #
    # # Check for required environment variables
    hf_token = os.getenv("HF_TOKEN")
    hf_dataset_repo = os.getenv("HF_DATASET_REPO")

    if not hf_token:
        print("Error: HF_TOKEN not found in .env file")
        return

    if not hf_dataset_repo:
        print("Error: HF_DATASET_REPO not found in .env file")
        return

    if not PATHS.zip_path.exists():
        print(f"Error: Zip file not found at {PATHS.zip_path}")
        return

    # version_path = BASE_DIR / "data" / "version.txt"
    # if not version_path.exists():
    #     print(f"Warning: version.txt not found at {version_path}")

    # Push to Hugging Face
    print(f"\nPushing to Hugging Face: {hf_dataset_repo}")
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(PATHS.zip_path),
            path_in_repo="YOLO_ka_words.zip",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            token=hf_token
        )

        # # Upload version.txt separately for easy version checking
        # if version_path.exists():
        #     api.upload_file(
        #         path_or_fileobj=str(version_path),
        #         path_in_repo="version.txt",
        #         repo_id=hf_dataset_repo,
        #         repo_type="dataset",
        #         token=hf_token
        #     )

        print(f"Successfully uploaded to https://huggingface.co/datasets/{hf_dataset_repo}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")


def main() -> None:
    generate_docs()
    # zip_dataset()
    # dataset_to_hf()


if __name__ == '__main__':
    main()
