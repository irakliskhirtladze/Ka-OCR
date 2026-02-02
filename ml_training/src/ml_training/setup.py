import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


def check_env() -> str:
    """Determines if running in Kaggle, Colab, or Local with high reliability."""
    # Kaggle-specific check (Primary)
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.path.exists('/kaggle/working'):
        return "kaggle"

    # Colab-specific check
    # Check for 'google.colab' module or unique Colab env var
    if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ:
        return "colab"

    return "local"


def download_from_hf(repo_id: str, filename: str, local_dir: Path, token: str = None, force: bool = False) -> Path:
    """Download a specific file from HuggingFace Hub."""
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=local_dir,
        force_download=force
    )
    return Path(path)


def dataset_needs_update(hf_repo: str, hf_token: str, local_version_path: Path) -> tuple[bool, str]:
    """
    Compares HF version with local version and checks if local dataset needs to be updated from HF.
    Returns tuple of (bool, hf_version string).
    """
    # Downloads HF version.txt to temp dir to avoid overwriting local
    with tempfile.TemporaryDirectory() as tmp_dir:
        hf_version_path = download_from_hf(hf_repo, "version.txt", Path(tmp_dir), hf_token, force=True)
        with open(hf_version_path, "r") as f:
            hf_version = f.read().strip()

    if not local_version_path.exists():
        return True, hf_version

    with open(local_version_path, "r") as f:
        local_version = f.read().strip()

    return hf_version > local_version, hf_version


@dataclass
class Paths:
    dataset_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    drive_dataset_zip_path: Path | None = None
    drive_output_dir: Path | None = None
    drive_checkpoint_dir: Path | None = None


def setup_environment() -> Paths:
    """
    Setup environment for model training depending on where the session is running.
    Returns dataclass instance with paths to dataset, checkpoints, outputs, etc.
    """
    env = check_env()
    print(f"running on {env}")

    if check_env() == "colab":  # set up colab session with google drive for permanent storage
        base_dir = Path("/content")
        paths = Paths(
            dataset_dir=base_dir / "data",
            output_dir=base_dir / "output",
            checkpoint_dir=base_dir / "checkpoints",
            drive_dataset_zip_path=base_dir / "drive/MyDrive/Colab Notebooks/trocr-ka/data/ka-ocr.zip",
            drive_output_dir=base_dir / "drive/MyDrive/Colab Notebooks/trocr-ka/output",
            drive_checkpoint_dir=base_dir / "drive/MyDrive/Colab Notebooks/trocr-ka/checkpoints",
        )

        # make sure session dirs exist. drive dirs are set up manually, no need for check
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not paths.dataset_dir.exists():
            print("\nExtracting dataset zip from Drive to session storage...")
            try:
                # -q "quiet", don't print filenames; -o overwrite existing files; -d specifies the destination dir
                cmd = ["unzip", "-o", "-q", str(paths.drive_dataset_zip_path), "-d", str(paths.dataset_dir)]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("Extraction successful!")
            except subprocess.CalledProcessError as e:
                print(f"Extraction failed!\nError: {e.stderr}")
        else:
            print("Dataset already copied to colab session.")

    elif check_env() == "kaggle":  # setup for kaggle
        base_dir = Path("/kaggle")
        paths = Paths(
            dataset_dir=base_dir / "input/ka-ocr",
            output_dir=base_dir / "working/output",
            checkpoint_dir=base_dir / "working/checkpoints",
        )
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    else:  # Setup for local session
        base_dir = Path(__file__).resolve().parent.parent.parent
        load_dotenv(base_dir / ".env")

        paths = Paths(
            dataset_dir=base_dir / "data",
            output_dir=base_dir / "output",
            checkpoint_dir=base_dir / "checkpoints"
        )

        paths.dataset_dir.mkdir(parents=True, exist_ok=True)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        hf_repo = os.getenv("HF_DATASET_REPO")
        hf_token = os.getenv("HF_TOKEN")

        print("Downloading version file from HuggingFace for comparison...")
        needs_update, hf_version = dataset_needs_update(hf_repo, hf_token, paths.dataset_dir / "version.txt")

        if needs_update:
            print(f"dataset needs update (HF version: {hf_version})")

            # Clear everything in local dataset dir
            print("Clearing old dataset...")
            for item in paths.dataset_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            # Download zip and extract
            print("downloading...")
            zip_path = download_from_hf(hf_repo, "ka-ocr.zip", paths.dataset_dir, hf_token, force=True)
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(paths.dataset_dir)

            # Delete zip
            zip_path.unlink()
            print("Extraction complete, zip deleted")

            # Redownload version.txt from HF
            download_from_hf(hf_repo, "version.txt", paths.dataset_dir, hf_token, force=True)

        print(f"Local dataset is the newest version already: {hf_version}. No update needed.")

    return paths
