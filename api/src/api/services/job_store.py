import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class Job:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["pending", "processing", "done", "failed"] = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    result_pdf: bytes | None = None
    error: str | None = None


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        job = Job()
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)
