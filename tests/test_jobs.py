"""Tests for app.jobs: in-memory job tracking for background batch uploads.

Run: python -m pytest tests/test_jobs.py -v

app.jobs keeps its state in a module-level dict for the life of the
process, so each test uses a fresh uuid-keyed job (create_job always
mints a new id) rather than needing to reset any shared state between
tests.
"""
from app import jobs


def test_create_job_returns_job_with_queued_files_in_order():
    job = jobs.create_job(["a.pdf", "b.pdf", "c.pdf"])

    assert job.status == "running"
    assert job.current_file_index == 0
    assert len(job.files) == 3
    assert [f.filename for f in job.files] == ["a.pdf", "b.pdf", "c.pdf"]
    assert all(f.stage == "queued" for f in job.files)
    assert all(f.error is None and f.result is None for f in job.files)
    assert job.created_at  # non-empty timestamp


def test_create_job_assigns_unique_ids():
    job1 = jobs.create_job(["a.pdf"])
    job2 = jobs.create_job(["a.pdf"])
    assert job1.id != job2.id


def test_get_job_returns_none_for_unknown_id():
    assert jobs.get_job("does-not-exist") is None


def test_get_job_returns_the_same_job_object_by_id():
    job = jobs.create_job(["a.pdf"])
    fetched = jobs.get_job(job.id)
    assert fetched is job


def test_set_current_file_updates_index():
    job = jobs.create_job(["a.pdf", "b.pdf"])
    jobs.set_current_file(job.id, 1)
    assert jobs.get_job(job.id).current_file_index == 1


def test_set_file_stage_updates_stage_and_detail():
    job = jobs.create_job(["a.pdf"])
    jobs.set_file_stage(job.id, 0, "extracting", "batch 2/5")

    fetched = jobs.get_job(job.id)
    assert fetched.files[0].stage == "extracting"
    assert fetched.files[0].stage_detail == "batch 2/5"


def test_set_file_stage_without_detail_clears_previous_detail():
    job = jobs.create_job(["a.pdf"])
    jobs.set_file_stage(job.id, 0, "extracting", "batch 1/3")
    jobs.set_file_stage(job.id, 0, "normalizing")

    fetched = jobs.get_job(job.id)
    assert fetched.files[0].stage == "normalizing"
    assert fetched.files[0].stage_detail is None


def test_set_file_result_marks_file_done_and_stores_result():
    job = jobs.create_job(["a.pdf"])
    result = {"document_id": 1, "facts_extracted": 3}
    jobs.set_file_result(job.id, 0, result)

    fetched = jobs.get_job(job.id)
    assert fetched.files[0].stage == "done"
    assert fetched.files[0].stage_detail is None
    assert fetched.files[0].result == result
    assert fetched.files[0].error is None


def test_set_file_error_marks_file_failed_with_message():
    job = jobs.create_job(["a.pdf"])
    jobs.set_file_error(job.id, 0, "Could not parse PDF: bad bytes")

    fetched = jobs.get_job(job.id)
    assert fetched.files[0].stage == "failed"
    assert fetched.files[0].error == "Could not parse PDF: bad bytes"
    assert fetched.files[0].result is None


def test_mutators_on_unknown_job_or_out_of_range_index_do_not_raise():
    jobs.set_current_file("nonexistent", 0)
    jobs.set_file_stage("nonexistent", 0, "parsing")
    jobs.set_file_result("nonexistent", 0, {})
    jobs.set_file_error("nonexistent", 0, "oops")
    jobs.mark_job_completed("nonexistent")
    jobs.mark_job_failed("nonexistent", "oops")

    job = jobs.create_job(["a.pdf"])
    jobs.set_file_stage(job.id, 5, "parsing")  # out of range index
    jobs.set_file_result(job.id, -1, {})
    assert jobs.get_job(job.id).files[0].stage == "queued"  # untouched


def test_mark_job_completed_sets_status_and_finished_at():
    job = jobs.create_job(["a.pdf"])
    assert job.finished_at is None

    jobs.mark_job_completed(job.id)

    fetched = jobs.get_job(job.id)
    assert fetched.status == "completed"
    assert fetched.finished_at is not None


def test_mark_job_failed_sets_status_error_and_finished_at():
    job = jobs.create_job(["a.pdf"])

    jobs.mark_job_failed(job.id, "unexpected boom")

    fetched = jobs.get_job(job.id)
    assert fetched.status == "failed"
    assert fetched.error == "unexpected boom"
    assert fetched.finished_at is not None


def test_any_running_reflects_current_state():
    job = jobs.create_job(["a.pdf"])
    assert jobs.any_running() is True

    jobs.mark_job_completed(job.id)
    # Other jobs from earlier tests may still be "running" in this
    # process-wide store, so only assert on this job's own contribution
    # by checking status directly, then re-verify any_running is still
    # correctly computed relative to a fresh job.
    assert jobs.get_job(job.id).status == "completed"

    fresh = jobs.create_job(["b.pdf"])
    assert jobs.any_running() is True
    jobs.mark_job_failed(fresh.id, "boom")
    assert jobs.get_job(fresh.id).status == "failed"


def test_any_running_false_when_no_jobs_running_at_all(monkeypatch):
    # Isolate from other tests' module-level state by swapping in a fresh
    # empty jobs dict for the duration of this test.
    monkeypatch.setattr(jobs, "_jobs", {})
    assert jobs.any_running() is False

    job = jobs.create_job(["a.pdf"])
    assert jobs.any_running() is True

    jobs.mark_job_completed(job.id)
    assert jobs.any_running() is False


def test_most_recent_running_job_id(monkeypatch):
    monkeypatch.setattr(jobs, "_jobs", {})
    assert jobs.most_recent_running_job_id() is None

    job1 = jobs.create_job(["a.pdf"])
    assert jobs.most_recent_running_job_id() == job1.id

    job2 = jobs.create_job(["b.pdf"])
    assert jobs.most_recent_running_job_id() == job2.id

    jobs.mark_job_completed(job2.id)
    assert jobs.most_recent_running_job_id() == job1.id

    jobs.mark_job_completed(job1.id)
    assert jobs.most_recent_running_job_id() is None


def test_job_status_payload_shape():
    job = jobs.create_job(["a.pdf", "b.pdf"])
    jobs.set_file_stage(job.id, 0, "extracting", "batch 1/2")
    jobs.set_file_result(job.id, 1, {"facts_extracted": 2})

    payload = jobs.job_status_payload(job)

    assert payload["job_id"] == job.id
    assert payload["status"] == "running"
    assert payload["total_files"] == 2
    assert payload["error"] is None
    assert payload["files"][0] == {
        "filename": "a.pdf",
        "stage": "extracting",
        "stage_detail": "batch 1/2",
        "error": None,
        "result": None,
    }
    assert payload["files"][1]["stage"] == "done"
    assert payload["files"][1]["result"] == {"facts_extracted": 2}
