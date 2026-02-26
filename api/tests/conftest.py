from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import jobs
from api.services.job_store import JobStore
from api.services.ocr import OCR, ImageItem, TextRegion


@pytest.fixture
def mock_ocr():
    ocr = MagicMock(spec=OCR)

    async def process_image(img_item: ImageItem) -> ImageItem:
        img_item.regions = [TextRegion(text="hello", x1=10, y1=10, x2=60, y2=30)]
        return img_item

    ocr.process_image = AsyncMock(side_effect=process_image)
    return ocr


@pytest.fixture
def app(mock_ocr):
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        app.state.ocr = mock_ocr
        app.state.job_store = JobStore()
        yield

    test_app = FastAPI(lifespan=test_lifespan)
    test_app.include_router(jobs.router, prefix="/v1")

    @test_app.get("/health")
    def health_check():
        return {"status": "online"}

    return test_app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
