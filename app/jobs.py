"""In-memory job tracking for background document-batch processing.

This is a prototype (per project convention: SQLite-only persistence, no
Redis/task queue), so job state lives in a plain module-level dict for the
lifetime of the process -- not persisted across restarts, and not safe for
multiple worker processes. That's fine for a single-process demo; this is
not meant to be mistaken for production job infrastructure.

Concurrency note: FastAPI's async routes and the `asyncio.create_task`
background jobs they kick off all run on the same single-threaded asyncio
event loop, so plain dict/list mutation here is safe without locks -- an
`await` is the only place control can switch to another coroutine, and
none of the mutator functions below contain one.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class FileProgress:
    filename: str
    # "queued" | "parsing" | "sectioning" | "extracting" | "normalizing"
    # | "storing" | "comparing" | "done" | "failed"
    stage: str = "queued"
    stage_detail: Optional[str] = None  # e.g. "batch 2/5"
    error: Optional[str] = None
    result: Optional[dict] = None  # the per-file result dict, once done


@dataclass
class Job:
    id: str
    status: str = "running"  # "running" | "completed" | "failed"
    files: list = field(default_factory=list)  # list[FileProgress], in order
    current_file_index: int = 0
    created_at: str = ""
    finished_at: Optional[str] = None
    error: Optional[str] = None  # set only if the whole job failed unexpectedly


_jobs: dict[str, Job] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(filenames: list[str]) -> Job:
    """Registers a new job with one FileProgress (stage="queued") per
    filename, in order, and returns it."""
    job = Job(
        id=uuid.uuid4().hex,
        status="running",
        files=[FileProgress(filename=fn) for fn in filenames],
        current_file_index=0,
        created_at=_now(),
    )
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def any_running() -> bool:
    """True if any tracked job is currently "running" -- used to show a
    banner elsewhere in the UI (e.g. the Facts page) when a batch is still
    in flight, so partial results don't look like a bug."""
    return any(job.status == "running" for job in _jobs.values())


def most_recent_running_job_id() -> Optional[str]:
    """Id of the most recently created still-running job, if any -- lets a
    "still processing" banner elsewhere link straight to that job's status
    page. Returns None if nothing is running."""
    running = [job for job in _jobs.values() if job.status == "running"]
    if not running:
        return None
    return max(running, key=lambda job: job.created_at).id


def set_current_file(job_id: str, index: int) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    job.current_file_index = index


def set_file_stage(job_id: str, index: int, stage: str, detail: Optional[str] = None) -> None:
    job = _jobs.get(job_id)
    if job is None or not (0 <= index < len(job.files)):
        return
    job.files[index].stage = stage
    job.files[index].stage_detail = detail


def set_file_result(job_id: str, index: int, result: dict) -> None:
    job = _jobs.get(job_id)
    if job is None or not (0 <= index < len(job.files)):
        return
    job.files[index].stage = "done"
    job.files[index].stage_detail = None
    job.files[index].result = result


def set_file_error(job_id: str, index: int, error: str) -> None:
    job = _jobs.get(job_id)
    if job is None or not (0 <= index < len(job.files)):
        return
    job.files[index].stage = "failed"
    job.files[index].stage_detail = None
    job.files[index].error = error


def mark_job_completed(job_id: str) -> None:
    job = _jobs.get(job_id)
    if job is None:
        return
    job.status = "completed"
    job.finished_at = _now()


def mark_job_failed(job_id: str, error: Optional[str] = None) -> None:
    """Marks a job failed outright -- used only when the background task's
    body raises something unexpected (i.e. a bug), not for the normal case
    of individual files failing (those are recorded per-file via
    set_file_error while the job itself still completes)."""
    job = _jobs.get(job_id)
    if job is None:
        return
    job.status = "failed"
    job.finished_at = _now()
    job.error = error


def job_status_payload(job: Job) -> dict:
    """JSON-serializable snapshot of a job's current state, shared by the
    /documents/batch/status/{job_id} API route and (indirectly, in spirit)
    the polling JS on the UI status page."""
    return {
        "job_id": job.id,
        "status": job.status,
        "current_file_index": job.current_file_index,
        "total_files": len(job.files),
        "error": job.error,
        "files": [
            {
                "filename": f.filename,
                "stage": f.stage,
                "stage_detail": f.stage_detail,
                "error": f.error,
                "result": f.result,
            }
            for f in job.files
        ],
    }
