import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from api.core.config import settings
from api.routes import jobs
from api.services.job_store import JobStore
from api.services.ocr import OCR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info(f"Loading TrOCR model: {settings.TROCR_MODEL_ID}")
    processor = TrOCRProcessor.from_pretrained(settings.TROCR_MODEL_ID)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(settings.TROCR_MODEL_ID).to(settings.DEVICE)

    logger.info(f"Loading YOLO model: {settings.YOLO_MODEL}")
    yolo_model = YOLO(settings.YOLO_MODEL, task="detect")

    app.state.ocr = OCR(yolo_model, trocr_model, processor, settings.DEVICE)
    app.state.job_store = JobStore()

    yield

    # SHUTDOWN
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)
app.include_router(jobs.router, prefix="/v1")


@app.get("/health")
def health_check() -> dict:
    return {"status": "online"}
