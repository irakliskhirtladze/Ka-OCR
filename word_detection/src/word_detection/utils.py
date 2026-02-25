import inspect
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    # local paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    dataset_dir: Path = base_dir / "dataset"
    train_dir: Path = dataset_dir / "train"
    val_dir: Path = dataset_dir / "val"
    zip_path: Path = dataset_dir / "YOLO_ka_words.zip"
    output_dir: Path = base_dir / "output"
    checkpoints_dir: Path = base_dir / "checkpoints"

    env_path: Path = base_dir / ".env"

    fonts_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "fonts"
    text_files_dir: Path = base_dir / "src" / "word_detection" / "data_factory" / "text_source"

    # drive paths
    drive_dataset_zip_path: Path = Path(
        "/content/drive/MyDrive/Colab Notebooks/word_detection/data/YOLO_ka_words.zip").resolve()
    drive_output_dir: Path = Path("/content/drive/MyDrive/Colab Notebooks/word_detection/output").resolve()
    drive_checkpoints_dir: Path = Path("/content/drive/MyDrive/Colab Notebooks/word_detection/checkpoints").resolve()

    def __post_init__(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    @property
    def yolo_project_dir(self) -> str:
        if check_env() == "colab":
            return str(self.drive_output_dir)
        return str(self.output_dir)


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
        if not any(PATHS.dataset_dir.iterdir()):
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

        if PATHS.drive_checkpoints_dir.exists():
            print("Syncing checkpoints from Drive to session...")
            # Use cp -r to bring previous work into the local working dir
            subprocess.run(["cp", "-r", f"{PATHS.drive_checkpoints_dir}/.", str(PATHS.checkpoints_dir)])

    else:  # Setup for local session
        pass


def sync_to_drive() -> None:
    """Call this to backup local outputs/checkpoints to Drive"""
    print("Backing up to Drive...")
    subprocess.run(["cp", "-r", f"{PATHS.output_dir}/.", str(PATHS.drive_output_dir)])
    subprocess.run(["cp", "-r", f"{PATHS.checkpoints_dir}/.", str(PATHS.drive_checkpoints_dir)])


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
