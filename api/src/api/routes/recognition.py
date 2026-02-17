import io
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from api.services.ocr import OCRService


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recognize")
async def recognize_text(request: Request, files: list[UploadFile] = File(...)) -> dict:
    results = []

    for file in files:
        if file.content_type == "application/pdf":
            # Logic for PDF
            pass
        elif file.content_type.startswith("image/"):
            # Logic for Image
            pass
        else:
            # Skip or Error
            continue

    return {"processed_files": len(files), "status": "success"}
