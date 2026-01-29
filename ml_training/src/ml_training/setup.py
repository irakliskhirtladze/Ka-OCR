import shutil
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import tempfile
from dataclasses import dataclass


def check_env() -> str:
    # Check for colab-specific environment variable
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
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
    Returns (needs_update, hf_version).
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
    env_vars_path: Path | None = None


def setup_environment() -> Paths:
    """
    Setup environment for model training depending on where the session is running.
    Returns paths dataclass.
    """
    env = check_env()
    print(f"running on {env}")

    if check_env() == "colab":
        # Here we setup colab session with google drive for permanent storage
        !pip install evaluate jiwer

        from evaluate import load
        from google.colab import drive
        drive.mount('/content/drive')

        base_dir = Path("/content")
        paths = Paths(
            dataset_dir=base_dir / "data",
            output_dir=base_dir / "output",
            checkpoint_dir=base_dir / "checkpoints",
            drive_dataset_zip_path=base_dir / "drive/MyDrive/Colab Notebooks/trocr-ka/data/ka-ocr.zip",
            drive_output_dir=base_dir / "drive/MyDrive/Colab Notebooks/trocr-ka/output",
            drive_checkpoint_dir=base_dir / "drive/MyDrive/Colab Notebooks/trocr-ka/checkpoints",
        )

    else:
        # Setup for local session
        base_dir = Path(__file__).resolve().parent.parent.parent
        load_dotenv(base_dir / ".env")

        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        hf_repo = os.getenv("HF_DATASET_REPO")
        hf_token = os.getenv("HF_TOKEN")

        if not hf_repo:
            raise ValueError("HF_DATASET_REPO not set in .env")

        local_version_path = data_dir / "version.txt"

        print("Downloading version file from HuggingFace for comparison...")
        should_download, hf_version = needs_download(hf_repo, hf_token, local_version_path)

        if should_download:
            print(f"dataset needs updating (HF version: {hf_version})")

            # Clear old data first
            print("Clearing old dataset...")
            for item in data_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            # Download zip and extract
            print("downloading...")
            zip_path = download_from_hf(hf_repo, "ka-ocr.zip", data_dir, hf_token, force=True)
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

            # Delete zip
            zip_path.unlink()
            print("Extraction complete, zip deleted")

            # Save the new version file locally
            with open(local_version_path, "w") as f:
                f.write(hf_version)

            return data_dir

        print(f"Local dataset is the newest version already: {hf_version}. No update needed.")

    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return paths
