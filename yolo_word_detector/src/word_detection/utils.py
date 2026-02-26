import inspect
import os
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    # local paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    dataset_dir: Path = base_dir / "dataset"
    real_docs_dir: Path = base_dir / "real_docs"
    train_dir: Path = dataset_dir / "train"
    val_dir: Path = dataset_dir / "val"
    zip_path: Path = dataset_dir / "YOLO_ka_words.zip"
    output_dir: Path = base_dir / "output"
    weights_dir: Path = output_dir / "ka_word_detector" / "weights"

    env_path: Path = base_dir / ".env"

    fonts_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "fonts"
    text_files_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "text_source"

    # drive paths
    drive_dataset_zip_path: Path = Path(
        "/content/drive/MyDrive/Colab Notebooks/word_detection/data/YOLO_ka_words.zip").resolve()
    drive_output_dir: Path = Path("/content/drive/MyDrive/Colab Notebooks/word_detection/output").resolve()

    def __post_init__(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def check_env() -> str:
    """Determines if running in Colab or locally."""
    if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ:
        return "colab"
    return "local"


def _dataset_is_populated() -> bool:
    """True if train_dir contains at least one image — guards against empty subdirs created by Paths init."""
    return any(PATHS.train_dir.glob("*.png"))


def _extract_dataset_zip(zip_path: Path) -> None:
    """Extract dataset zip into dataset_dir, stripping the top-level YOLO_ka_words/ prefix."""
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for member in archive.namelist():
            try:
                relative = Path(member).relative_to("YOLO_ka_words")
            except ValueError:
                continue
            target = PATHS.dataset_dir / relative
            if member.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))


def setup_environment() -> None:
    """Setup session for model training; handles dataset extraction and checkpoint sync from Drive."""
    env = check_env()
    print(f"Running on: {env}")

    if env == "colab":
        if not _dataset_is_populated():
            if not PATHS.drive_dataset_zip_path.exists():
                raise FileNotFoundError(f"Dataset zip not found on Drive: {PATHS.drive_dataset_zip_path}")
            print(f"Extracting dataset from Drive...")
            _extract_dataset_zip(PATHS.drive_dataset_zip_path)
            print("Extraction complete.")
        else:
            print("Dataset already present in session storage.")

        if PATHS.drive_output_dir.exists():
            print("Restoring previous training output from Drive...")
            subprocess.run(["cp", "-r", f"{PATHS.drive_output_dir}/.", str(PATHS.output_dir)], check=True)


def sync_to_drive() -> None:
    """Backup local training output to Drive."""
    PATHS.drive_output_dir.mkdir(parents=True, exist_ok=True)
    print("Syncing to Drive...")
    subprocess.run(["cp", "-r", f"{PATHS.output_dir}/.", str(PATHS.drive_output_dir)], check=True)


def create_yaml() -> None:
    # cleandoc removes the leading spaces caused by the function's indentation
    yaml_content = inspect.cleandoc(f"""
        path: {str(PATHS.dataset_dir.absolute())}
        train: train
        val: val

        names:
          0: word
    """)

    with open(PATHS.base_dir / "ka_dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"YAML created successfully at {PATHS.base_dir / 'ka_dataset.yaml'}")
