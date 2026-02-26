import io

import numpy as np
import pymupdf
import pytest

from api.services.ocr import ImageItem, TextRegion
from api.services.pdf import build_searchable_pdf, pdf_to_images


def make_pdf_bytes(pages: int = 1) -> bytes:
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page(width=200, height=100)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_image(h: int = 100, w: int = 200) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 200


def test_pdf_to_images_count():
    images = pdf_to_images(make_pdf_bytes(pages=1))
    assert len(images) == 1


def test_pdf_to_images_multi_page_count():
    images = pdf_to_images(make_pdf_bytes(pages=3))
    assert len(images) == 3


def test_pdf_to_images_returns_numpy_arrays():
    images = pdf_to_images(make_pdf_bytes())
    assert isinstance(images[0], np.ndarray)
    assert images[0].ndim == 3
    assert images[0].shape[2] == 3  # RGB


def test_build_searchable_pdf_returns_bytes():
    img_item = ImageItem(img_id="t", img=make_image(), regions=[])
    result = build_searchable_pdf([img_item])
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_build_searchable_pdf_is_valid_pdf():
    regions = [TextRegion(text="hello", x1=10, y1=10, x2=60, y2=30)]
    img_item = ImageItem(img_id="t", img=make_image(), regions=regions)
    pdf_bytes = build_searchable_pdf([img_item])
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert doc.page_count == 1
    doc.close()


def test_build_searchable_pdf_page_count():
    img_items = [ImageItem(img_id=f"p{i}", img=make_image(), regions=[]) for i in range(3)]
    pdf_bytes = build_searchable_pdf(img_items)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert doc.page_count == 3
    doc.close()


def test_build_searchable_pdf_empty_regions():
    img_item = ImageItem(img_id="t", img=make_image(), regions=[])
    result = build_searchable_pdf([img_item])
    assert isinstance(result, bytes)
