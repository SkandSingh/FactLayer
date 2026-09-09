## Setup and Run Instructions

Requires Python 3.11+ and at least one [Gemini API key](https://aistudio.google.com/apikey)
(the app uses `gemini-flash-lite-latest` by default — it has a workable free tier,
though a fresh free-tier key can be capped as low as 5 requests/minute).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your-key-here
```

**Multiple keys / providers (recommended):** every LLM call round-robins
across every key you configure, so each additional key adds its own
rate-limit budget to the pool — this is what actually fixes rate-limit
stalls, not just retries. In `.env`:
```bash
GEMINI_API_KEYS=key1,key2
GROQ_API_KEYS=key1,key2   # optional, Groq (groq.com) — same LLMClient interface
```
(`GEMINI_API_KEY`, singular, still works as a one-key shorthand.)

```bash
uvicorn app.main:app --reload
```

The app is now running at `http://localhost:8000`:

- `http://localhost:8000/ui/upload` — upload one or more PDFs from the
  browser; submitting redirects to a live progress page that polls and
  shows each file's current stage (parsing → sectioning → extracting →
  normalizing → storing → comparing) until it's done
- `http://localhost:8000/ui` — the fact browser (uploaded documents and
  their facts show up here; shows a banner if a batch is still processing)
- `http://localhost:8000/docs` — interactive API docs (Swagger UI), if you'd
  rather upload via the API directly
- `http://localhost:8000/health` — health check

To upload a PDF from the command line instead:

```bash
curl -X POST http://localhost:8000/documents -F "file=@/path/to/your.pdf"

# or several at once — each is compared against the whole existing store,
# including the others in this same batch, as they're processed in order:
curl -X POST http://localhost:8000/documents/batch \
  -F "files=@one.pdf" -F "files=@two.pdf" -F "files=@three.pdf"
```

`GET /stats` gives the running totals (documents, facts, relationships,
broken down by corroborates/contradicts/reconciled_context) across
everything ingested so far.

Run the test suite with:

```bash
pytest tests/
```
