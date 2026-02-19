import itertools

import cv2
from PIL import Image
from random import randint, choice

from word_detection.data_factory.doc_gen import DocumentGenerator
from word_detection.utils import PATHS


def main() -> None:
    texts: list[str] = []
    for text_file_path in PATHS.text_files_dir.glob("*.txt"):
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            texts.append(text_file.read())

    font_files = itertools.chain(
        PATHS.fonts_dir.glob("*.ttf"),
        PATHS.fonts_dir.glob("*.otf")
    )
    for font_path in font_files:
        for i in range(2):
            file_name = f"{str(font_path.stem)}-{i}"
            doc_generator = DocumentGenerator(
                file_name,
                randint(640, 2000),
                randint(640, 2000),
                font_path,
                randint(11, 28),
                choice(texts)
            )

            img, bboxes = doc_generator.render_document(mode=choice(["table", "block"]))

            for bbox in bboxes:
                cls, cx, cy, w, h = bbox
                x1 = int((cx - w / 2) * doc_generator.width)
                y1 = int((cy - h / 2) * doc_generator.height)
                x2 = int((cx + w / 2) * doc_generator.width)
                y2 = int((cy + h / 2) * doc_generator.height)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            cv2.imwrite(str(PATHS.dataset_dir / file_name)+".png", img)
            with open(str(PATHS.dataset_dir / file_name) + ".txt", "w", encoding="utf-8") as f:
                for bbox in bboxes:
                    cls, cx, cy, w, h = bbox
                    f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


if __name__ == '__main__':
    main()
