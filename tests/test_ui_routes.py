"""Tests for the server-rendered /ui routes: index, fact detail, relationship detail.

Run: python -m pytest tests/test_ui_routes.py -v

Uses the same temp_db isolation pattern as test_store.py / test_facts_route.py /
test_relationships_route.py: monkeypatch config.FACTLAYER_DB_PATH to a tmp_path
file, then insert test data directly via store functions before hitting the
HTTP endpoints through TestClient.
"""
import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from app import config, jobs, store
from app.llm.base import LLMClient
from app.main import app
from app.models import ExtractedFact, TimeScope
from app.routes import documents as documents_routes

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")


class FakeLLMClient(LLMClient):
    """Test double for LLMClient (same pattern as test_documents_route.py)."""

    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._response


def _canned_fact_dict() -> dict:
    return {
        "entity_name": "Acme Corp",
        "entity_id": None,
        "attribute": "revenue_from_operations",
        "raw_value": "100",
        "unit": "Cr",
        "time_scope": {
            "period_type": "point_in_time",
            "start": None,
            "end": None,
            "label": "FY24",
            "vintage": None,
        },
        "entity_scope": "standalone",
        "verbatim_quote": "some quote that may or may not appear verbatim",
        "confidence": 0.9,
    }


def make_extracted_fact(**overrides) -> ExtractedFact:
    """Factory for creating ExtractedFact test data."""
    defaults = dict(
        entity_name="Delhivery Limited",
        entity_id="L63090HR2011PLC044150",
        attribute="revenue_from_operations",
        raw_value="2,067.6",
        unit="Rs Million",
        time_scope=TimeScope(
            period_type="span",
            start="2023-04-01",
            end="2024-03-31",
            label="FY24",
            vintage=None,
        ),
        entity_scope="consolidated",
        verbatim_quote="Revenue from operations was Rs 2,067.6 Million in FY24.",
        confidence=0.95,
    )
    defaults.update(overrides)
    return ExtractedFact(**defaults)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Isolate every test in this file from the real factlayer.db."""
    db_path = tmp_path / "test_factlayer.db"
    monkeypatch.setattr(config, "FACTLAYER_DB_PATH", str(db_path))
    yield db_path


@pytest.fixture
def client():
    # Context-manager form keeps the TestClient's anyio blocking portal (and
    # its event loop) alive across requests within a test -- required for
    # the background asyncio.create_task behind /ui/upload's job to still
    # be running by the time a later request polls its status. A bare
    # TestClient(app) spins up and tears down a fresh portal per request,
    # which would cancel that task before it progresses at all.
    with TestClient(app) as c:
        yield c


def _poll_job_until_finished(client, job_id, timeout_seconds=2.0, interval_seconds=0.05):
    """See tests/test_documents_route.py's helper of the same name -- the
    background job only makes progress as the test yields control (via
    each HTTP call TestClient makes), so this polls with a bounded retry
    loop rather than relying on a single check or a fixed sleep. Polls the
    JSON status endpoint (shared by both entry points) rather than
    re-fetching/re-parsing the HTML status page each time.
    """
    deadline = time.monotonic() + timeout_seconds
    body = None
    while time.monotonic() < deadline:
        resp = client.get(f"/documents/batch/status/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(interval_seconds)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_seconds}s: {body}")


def test_index_returns_200_and_contains_seeded_data(client):
    """GET /ui/ returns 200 and shows seeded document + fact data."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 3)
    store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(attribute="revenue_from_operations"),
        section_path="Financial Statements > Revenue",
        char_offset=120,
        page_number=4,
    )

    resp = client.get("/ui/")
    assert resp.status_code == 200
    body = resp.text
    assert "Delhivery Limited" in body
    assert "revenue_from_operations" in body


def test_fact_detail_returns_200_and_contains_verbatim_quote(client):
    """GET /ui/facts/{id} for an existing fact returns 200 with its verbatim_quote."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 3)
    fact_id = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(),
        section_path="Financial Statements > Revenue",
        char_offset=120,
        page_number=4,
    )

    resp = client.get(f"/ui/facts/{fact_id}")
    assert resp.status_code == 200
    assert "Revenue from operations was Rs 2,067.6 Million in FY24." in resp.text


def test_fact_detail_for_nonexistent_id_returns_404(client):
    """GET /ui/facts/{id} for a nonexistent id returns 404."""
    resp = client.get("/ui/facts/9999")
    assert resp.status_code == 404


def test_relationship_detail_returns_200_and_contains_both_facts(client):
    """GET /ui/relationships/{id} shows both facts' entity_name and verbatim_quote."""
    doc_id = store.insert_document("delhivery_fy24.pdf", "Delhivery Limited", 3)
    fact_a = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Delhivery Limited",
            attribute="revenue_from_operations",
            verbatim_quote="Revenue from operations was Rs 2,067.6 Million in FY24.",
        ),
        section_path="Financial Statements > Revenue",
        char_offset=120,
        page_number=4,
    )
    fact_b = store.insert_fact(
        document_id=doc_id,
        fact=make_extracted_fact(
            entity_name="Zomato Limited",
            attribute="total_revenue",
            verbatim_quote="Total revenue reported was Rs 2,060 Million for FY24.",
        ),
        section_path="Notes to Accounts > Revenue",
        char_offset=340,
        page_number=7,
    )
    rel_id = store.insert_relationship(
        fact_id_a=fact_a,
        fact_id_b=fact_b,
        relation_type="corroborates",
        reconciled_dimension=None,
        reasoning_text="Both figures are consistent within rounding.",
        confidence=0.88,
    )

    resp = client.get(f"/ui/relationships/{rel_id}")
    assert resp.status_code == 200
    body = resp.text
    assert "Delhivery Limited" in body
    assert "Zomato Limited" in body
    assert "Revenue from operations was Rs 2,067.6 Million in FY24." in body
    assert "Total revenue reported was Rs 2,060 Million for FY24." in body


def test_relationship_detail_for_nonexistent_id_returns_404(client):
    """GET /ui/relationships/{id} for a nonexistent id returns 404."""
    resp = client.get("/ui/relationships/9999")
    assert resp.status_code == 404


def test_upload_page_get_returns_200_with_form(client):
    """GET /ui/upload shows the upload form."""
    resp = client.get("/ui/upload")
    assert resp.status_code == 200
    assert "form" in resp.text.lower()
    assert 'name="files"' in resp.text


def test_upload_page_post_redirects_to_status_page(client):
    """POST /ui/upload no longer processes inline: it kicks off a
    background job and redirects (303, so a refresh doesn't resubmit the
    form) to /ui/upload/status/{job_id}."""
    canned_response = json.dumps([_canned_fact_dict()])
    fake_client = FakeLLMClient(response=canned_response)
    app.dependency_overrides[documents_routes.get_llm_client] = lambda: fake_client

    try:
        with open(FIXTURE_PATH, "rb") as f:
            resp = client.post(
                "/ui/upload",
                files={"files": ("sample.pdf", f, "application/pdf")},
                follow_redirects=False,
            )

        assert resp.status_code == 303
        location = resp.headers["location"]
        assert location.startswith("/ui/upload/status/")
        job_id = location.rsplit("/", 1)[-1]

        final = _poll_job_until_finished(client, job_id)
    finally:
        app.dependency_overrides.pop(documents_routes.get_llm_client, None)

    assert final["status"] == "completed"
    assert len(final["files"]) == 1
    assert final["files"][0]["result"]["entity_name"] == "Acme Corp"

    persisted_docs = store.list_documents()
    assert len(persisted_docs) == 1
    assert persisted_docs[0].filename == "sample.pdf"

    # And the status page itself renders the completed state without JS.
    status_resp = client.get(f"/ui/upload/status/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.text
    assert "sample.pdf" in body
    assert "Acme Corp" in body
    assert fake_client.call_count > 0


def test_upload_page_post_rejects_non_pdf_without_crashing(client):
    """A non-PDF upload through the UI form still redirects to a status
    page (not a 500), and that page ends up showing the file as failed
    with a clear "must be a PDF" error rather than silently dropping it."""
    resp = client.post(
        "/ui/upload",
        files={"files": ("notes.txt", b"not a pdf", "text/plain")},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    job_id = resp.headers["location"].rsplit("/", 1)[-1]

    final = _poll_job_until_finished(client, job_id)
    assert final["status"] == "completed"
    assert final["files"][0]["stage"] == "failed"
    assert "must be a PDF" in final["files"][0]["error"]

    status_resp = client.get(f"/ui/upload/status/{job_id}")
    assert status_resp.status_code == 200
    assert "must be a PDF" in status_resp.text
    assert store.list_documents() == []


def test_upload_status_page_for_unknown_job_returns_404(client):
    resp = client.get("/ui/upload/status/does-not-exist")
    assert resp.status_code == 404


def test_index_shows_still_processing_banner_when_job_running(client):
    """GET /ui/ shows a clear banner when a batch job is still running, so
    partial results on the Facts page don't read like a bug."""
    job = jobs.create_job(["still_running.pdf"])

    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "still processing" in resp.text.lower()
    assert f"/ui/upload/status/{job.id}" in resp.text

    jobs.mark_job_completed(job.id)


def test_index_does_not_show_banner_when_no_job_running(client, monkeypatch):
    # Isolate from any running jobs left behind by other tests in this
    # process (app.jobs is a module-level, process-wide store).
    monkeypatch.setattr(jobs, "_jobs", {})

    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "still processing" not in resp.text.lower()
