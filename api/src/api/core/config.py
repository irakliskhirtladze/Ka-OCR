import os
from pathlib import Path

import torch


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    PROJECT_NAME: str = "ka-OCR"
    TROCR_MODEL_ID: str = os.getenv("TROCR_MODEL_ID", str(BASE_DIR / "models/trocr-ka"))
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", str(BASE_DIR / "models/yolo/yolo26n.pt"))
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PORT: int = int(os.getenv("PORT", 8000))  # GCP often passes a PORT env var, we can capture it here if needed


settings = Settings()
