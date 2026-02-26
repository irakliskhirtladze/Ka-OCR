from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest

from api.services.ocr import OCR, ImageItem, TextRegion


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.ones((100, 200, 3), dtype=np.uint8) * 128


def make_yolo_mock(boxes: list[list[float]]) -> MagicMock:
    """Creates a YOLO mock that returns the given list of [x1,y1,x2,y2] boxes."""
    mock_boxes = []
    for coords in boxes:
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = coords
        mock_boxes.append(box)

    result = MagicMock()
    result.boxes = mock_boxes

    yolo = MagicMock()
    yolo.predict.return_value = [result]
    return yolo


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.batch_decode.return_value = ["hello"]
    return processor


@pytest.fixture
def mock_trocr():
    return MagicMock()


@pytest.fixture
def ocr(mock_processor, mock_trocr, sample_image) -> OCR:
    yolo = make_yolo_mock([[10.0, 20.0, 60.0, 40.0]])
    return OCR(yolo, mock_trocr, mock_processor, "cpu")


def test_detect_bboxes_populates_regions(ocr, sample_image):
    img_item = ImageItem(img_id="t", img=sample_image)
    result = ocr.detect_bboxes(img_item)
    assert len(result.regions) == 1
    r = result.regions[0]
    assert (r.x1, r.y1, r.x2, r.y2) == (10, 20, 60, 40)


def test_detect_bboxes_sorts_by_position(sample_image, mock_trocr, mock_processor):
    # Two boxes: lower one first in the mock output → must be sorted to second
    yolo = make_yolo_mock([[100.0, 50.0, 150.0, 70.0], [10.0, 10.0, 60.0, 30.0]])
    ocr = OCR(yolo, mock_trocr, mock_processor, "cpu")
    result = ocr.detect_bboxes(ImageItem(img_id="t", img=sample_image))
    assert result.regions[0].y1 < result.regions[1].y1


def test_crop_region_shape(sample_image):
    region = TextRegion(text="", x1=10, y1=20, x2=60, y2=40)
    cropped = OCR.crop_region(sample_image, region)
    assert cropped.shape == (20, 50, 3)  # height=40-20, width=60-10


def test_image_item_is_null_when_no_regions(sample_image):
    assert ImageItem(img_id="t", img=sample_image).is_null() is True


def test_image_item_is_not_null_with_regions(sample_image):
    region = TextRegion(text="hi", x1=0, y1=0, x2=10, y2=10)
    assert ImageItem(img_id="t", img=sample_image, regions=[region]).is_null() is False


async def test_process_image(ocr, sample_image):
    result = await ocr.process_image(ImageItem(img_id="t", img=sample_image))
    assert len(result.regions) == 1
    assert result.regions[0].text == "hello"
