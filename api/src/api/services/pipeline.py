import io
import logging

import numpy as np
from PIL import Image

from api.services.job_store import Job, JobStore
from api.services.ocr import OCR, ImageItem
from api.services.pdf import build_searchable_pdf, pdf_to_images

logger = logging.getLogger(__name__)


async def run_pipeline(
    job: Job,
    job_store: JobStore,
    ocr: OCR,
    files: list[tuple[str, str, bytes]],  # (filename, content_type, file_bytes)
) -> None:
    job.status = "processing"
    try:
        img_items = []
        for filename, content_type, file_bytes in files:
            if content_type == "application/pdf":
                raw_images = pdf_to_images(file_bytes)
            elif content_type.startswith("image/"):
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                raw_images = [np.array(pil_img)]
            else:
                logger.warning(f"Skipping unsupported file: {filename} ({content_type})")
                continue

            for i, img in enumerate(raw_images):
                img_item = await ocr.process_image(ImageItem(img_id=f"{filename}_p{i}", img=img))
                img_items.append(img_item)

        job.result_pdf = build_searchable_pdf(img_items)
        job.status = "done"
        logger.info(f"Job {job.job_id} completed. Pages processed: {len(img_items)}")
    except Exception as e:
        logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
        job.status = "failed"
        job.error = str(e)