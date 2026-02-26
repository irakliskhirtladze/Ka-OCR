import io

import cv2
import numpy as np
import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def make_png_bytes() -> bytes:
    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def make_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=200, height=100)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_health_check(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "online"}


def test_submit_job_returns_202(client: TestClient):
    resp = client.post("/v1/jobs/", files=[("files", ("test.png", make_png_bytes(), "image/png"))])
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_submit_pdf_job_returns_202(client: TestClient):
    resp = client.post("/v1/jobs/", files=[("files", ("test.pdf", make_pdf_bytes(), "application/pdf"))])
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_get_job_status(client: TestClient):
    post = client.post("/v1/jobs/", files=[("files", ("test.png", make_png_bytes(), "image/png"))])
    job_id = post.json()["job_id"]

    resp = client.get(f"/v1/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] in ("pending", "processing", "done", "failed")


def test_get_job_not_found(client: TestClient):
    resp = client.get("/v1/jobs/nonexistent-id")
    assert resp.status_code == 404


def test_download_not_found(client: TestClient):
    resp = client.get("/v1/jobs/nonexistent-id/download")
    assert resp.status_code == 404


def test_download_returns_pdf_when_done(client: TestClient):
    post = client.post("/v1/jobs/", files=[("files", ("test.png", make_png_bytes(), "image/png"))])
    job_id = post.json()["job_id"]

    status = client.get(f"/v1/jobs/{job_id}").json()["status"]
    if status == "done":
        dl = client.get(f"/v1/jobs/{job_id}/download")
        assert dl.status_code == 200
        assert "application/pdf" in dl.headers["content-type"]


def test_download_pending_job_returns_409(client: TestClient, app: FastAPI):
    # Inject a job that is still pending directly into the store
    job = app.state.job_store.create()
    # status stays "pending" (default)

    resp = client.get(f"/v1/jobs/{job.job_id}/download")
    assert resp.status_code == 409


def test_submit_multiple_files(client: TestClient):
    resp = client.post(
        "/v1/jobs/",
        files=[
            ("files", ("a.png", make_png_bytes(), "image/png")),
            ("files", ("b.png", make_png_bytes(), "image/png")),
        ],
    )
    assert resp.status_code == 202
