import io
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
from api.services.ocr import OCRService


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recognize")
async def recognize_text(request: Request, file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get the service from the app state (set in lifespan)
        ocr_service: OCRService = request.app.state.ocr_service
        recognized_text = await ocr_service.recognize(image)

        return {
            "recognized_text": recognized_text,
            "filename": file.filename,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"OCR failure: {e}")
        raise HTTPException(500, detail=str(e))
