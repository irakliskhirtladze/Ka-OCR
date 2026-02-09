from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent


@dataclass
class Paths:
    raw_dir: Path
    zip_path: Path
    metadata_path: Path
    version_txt_path: Path
    real_augmented_dir: Path
    real_augmented_csv_path: Path


