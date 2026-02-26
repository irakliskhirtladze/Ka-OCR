from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent


@dataclass
class Paths:
    dict_path: Path = BASE_DIR / "src" / "dataset_gen" / "dictionaries" / "ka_dictionary.json"
    ka_font_dir: Path = BASE_DIR / "src" / "dataset_gen" / "generator" / "fonts" / "ka"

    synthetic_dir: Path = BASE_DIR / "data" / "synthetic"
    synthetic_metadata_path: Path = BASE_DIR / "data" / "synthetic" / "metadata.csv"

    real_images_dir: Path = BASE_DIR / "data" / "real"
    real_json_path: Path = BASE_DIR / "data" / "real" / "real.json"
    real_metadata_path: Path = BASE_DIR / "data" / "real" / "metadata.csv"

    zip_path: Path = BASE_DIR / "data" / "ka-ocr.zip"
    version_txt_path: Path = BASE_DIR / "data" / "version.txt"


PATHS = Paths()


# paths = Paths(
#         dict_path=BASE_DIR / "generator" / "dictionaries" / "ka_dictionary.json",
#         ka_font_dir=BASE_DIR / "src" / "dataset_gen" / "generator" / "fonts" / "ka",
#         synthetic_dir=BASE_DIR / "data" / "synthetic",
#         synthetic_metadata_path=BASE_DIR / "data" / "synthetic" / "metadata.csv",
#
#         real_images_dir=BASE_DIR / "data" / "real",
#         real_json_path=BASE_DIR / "data" / "real" / "real.json",
#         real_metadata_path=BASE_DIR / "data" / "real" / "metadata.csv",
#
#         zip_path=BASE_DIR / "data" / "ka-ocr.zip",
#         version_txt_path=BASE_DIR / "data" / "version.txt",
#     )

