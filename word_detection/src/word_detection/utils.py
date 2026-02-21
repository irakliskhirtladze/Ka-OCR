from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    dataset_dir: Path = base_dir / "dataset"
    zip_path: Path = dataset_dir / "YOLO_ka_words.zip"

    env_path: Path = base_dir / ".env"

    fonts_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "fonts"
    text_files_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "text_source"

    def __post_init__(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)


PATHS = Paths()
