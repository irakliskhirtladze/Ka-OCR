import logging

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from api.schemas.job import JobCreatedResponse, JobStatusResponse
from api.services.job_store import JobStore
from api.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/", response_model=JobCreatedResponse, status_code=202)
async def submit_job(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> JobCreatedResponse:
    job_store: JobStore = request.app.state.job_store
    job = job_store.create()

    # Read all file bytes before the response is sent — UploadFile is not safe to
    # access from a background task after the request scope closes.
    file_data = [
        (f.filename or "upload", f.content_type or "", await f.read())
        for f in files
    ]
    background_tasks.add_task(run_pipeline, job, job_store, request.app.state.ocr, file_data)

    return JobCreatedResponse(job_id=job.job_id, status=job.status)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    job_store: JobStore = request.app.state.job_store
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.job_id, status=job.status, error=job.error)


@router.get("/{job_id}/download")
async def download_result(job_id: str, request: Request) -> Response:
    job_store: JobStore = request.app.state.job_store
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready. Current status: {job.status}")
    return Response(
        content=job.result_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=ocr_{job_id}.pdf"},
    )
