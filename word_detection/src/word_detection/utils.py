import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    dataset_dir: Path = base_dir / "dataset"
    zip_path: Path = dataset_dir / "YOLO_ka_words.zip"
    output_dir: Path = base_dir / "output"
    checkpoints_dir: Path = base_dir / "checkpoints"

    drive_dataset_zip_path: Path = base_dir / "drive/MyDrive/Colab Notebooks/word_detection/data/YOLO_ka_words.zip"
    drive_output_dir: Path = base_dir / "drive/MyDrive/Colab Notebooks/word_detection/output",
    drive_checkpoints_dir: Path = base_dir / "drive/MyDrive/Colab Notebooks/word_detection/checkpoints",

    env_path: Path = base_dir / ".env"

    fonts_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "fonts"
    text_files_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "text_source"

    def __post_init__(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def check_env() -> str:
    """Determines if running in Colab, or Locally"""
    # Colab-specific check
    # Check for 'google.colab' module or unique Colab env var
    if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ:
        return "colab"

    return "local"


def setup_environment() -> None:
    """
    Setup environment for model training depending on where the session is running.
    Returns dataclass instance with paths to dataset, checkpoints, outputs, etc.
    """
    env = check_env()
    print(f"running on {env}")

    if check_env() == "colab":  # set up colab session with google drive for permanent storage
        print(f"base dir: {PATHS.base_dir}")
        if len(list(PATHS.dataset_dir.iterdir())) <= 0:
            print("\nExtracting dataset zip from Drive to session storage...")
            try:
                # -j: junk paths (do not create internal folders);-q "quiet", don't print filenames;
                # -o overwrite existing files; -d specifies the destination dir
                cmd = ["unzip", "-j", "-o", "-q", str(PATHS.drive_dataset_zip_path), "-d", str(PATHS.dataset_dir)]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("Extraction successful!")
            except subprocess.CalledProcessError as e:
                print(f"Extraction failed!\nError: {e.stderr}")
        else:
            print("Dataset already copied to colab session.")

    else:  # Setup for local session
        pass
