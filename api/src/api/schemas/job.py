from typing import Literal

from pydantic import BaseModel


class JobCreatedResponse(BaseModel):
    job_id: str
    status: Literal["pending"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "done", "failed"]
    error: str | None = None
