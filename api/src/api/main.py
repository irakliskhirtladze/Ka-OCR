import logging
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from api.core.config import settings
from api.routes import recognition
from api.services.ocr import OCRService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info(f"Loading model: {settings.MODEL_ID}")

    processor = TrOCRProcessor.from_pretrained(settings.MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(settings.MODEL_ID)

    if torch.cuda.is_available():
        model.to("cuda")

    # Store the Service in state so routes can access it
    app.state.ocr_service = OCRService(processor, model)

    yield

    # SHUTDOWN
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)
app.include_router(recognition.router, prefix="/v1")


@app.get("/")
def health_check() -> dict:
    return {"status": "online"}
