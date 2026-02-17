import os
from pathlib import Path


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    PROJECT_NAME: str = "TrOCR-ka"
    MODEL_ID: str = os.getenv("MODEL_ID", str(BASE_DIR / "models/trocr-ka"))
    PORT: int = int(os.getenv("PORT", 8000))  # GCP often passes a PORT env var, we can capture it here if needed


settings = Settings()
