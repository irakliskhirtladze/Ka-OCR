import io

import cv2
import numpy as np
import pymupdf

from api.services.ocr import ImageItem


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[np.ndarray]:
    """Renders each page of a PDF as an RGB numpy array."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    scale = dpi / 72  # pymupdf default resolution is 72 dpi
    mat = pymupdf.Matrix(scale, scale)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        images.append(img.copy())
    doc.close()
    return images


def build_searchable_pdf(img_items: list[ImageItem]) -> bytes:
    """
    Creates a searchable PDF: each page is the source image with
    invisible OCR text overlaid at the correct bbox positions.
    """
    doc = pymupdf.open()
    for item in img_items:
        h, w = item.img.shape[:2]
        page = doc.new_page(width=w, height=h)

        # Embed source image as page background
        _, png_buf = cv2.imencode(".png", cv2.cvtColor(item.img, cv2.COLOR_RGB2BGR))
        page.insert_image(pymupdf.Rect(0, 0, w, h), stream=png_buf.tobytes())

        # Overlay invisible text for each recognized word (render_mode=3 = invisible in PDF spec)
        for region in item.regions:
            if not region.text:
                continue
            rect = pymupdf.Rect(region.x1, region.y1, region.x2, region.y2)
            fontsize = max(6, region.y2 - region.y1)
            page.insert_textbox(rect, region.text, fontsize=fontsize, render_mode=3)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()
