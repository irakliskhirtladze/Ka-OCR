from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    dataset_dir: Path = base_dir / "dataset"
    source_docs_dir: Path = dataset_dir / "source_docs"

    def __post_init__(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

