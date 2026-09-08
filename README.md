# FactLayer

A Fact Knowledge Layer: upload PDFs, and it extracts factual claims, grounds
each one in the exact source location it came from, and flags how facts
relate across documents — corroborating, contradicting, or reconciled by
context (time, scope, units, or estimate vintage).

Built for the Superjoin VIT 2026 Engineering Intern hiring assignment.

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
  browser and see the extraction results
- `http://localhost:8000/ui` — the fact browser (uploaded documents and
  their facts show up here)
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

## Video Demo

*[Add your ≤3-minute demo video link here before submitting — showing a PDF
being uploaded via `/docs` or `/ui`, and the four required cases below
visible in the fact browser / relationship detail pages.]*

## Approach

**Pipeline (see the six stages below).** Code does everything deterministic;
the LLM only does the two things that genuinely require judgment — reading
what a section of text asserts, and deciding whether two facts agree,
conflict, or are both true under different conditions.

```
PDF ──▶ parse ──▶ section ──▶ extract (LLM) ──▶ normalize (code)
                                                       │
                                                       ▼
                                       compare (LLM, digest-style)
                                                       │
                                                       ▼
                                          store + API + UI
```

1. **Parse** (`app/pdf_ingest.py`) — PyMuPDF reads each page's text plus
   per-span layout metadata (font size, bold, bbox). Detected tables
   (`page.find_tables()`) are serialized as markdown tables in place of
   their raw flattened cell text, so row/column alignment survives into
   what the LLM reads — a plain span-by-position flatten scrambles which
   number belongs to which row label, which matters a lot for financial
   statements. Every span (table or not) carries a pointer back to
   `(page_number, char_offset)` into that page's plain text — this is the
   entire evidence story, captured at parse time rather than reconstructed
   later by re-searching the PDF.

2. **Section** (`app/sectioning.py`) — detects heading boundaries from
   generic layout signals (a span notably larger or bolder than the
   surrounding body text), not any document-specific heading list, so it
   generalizes to PDFs it's never seen. Falls back to fixed ~10-page
   groups when a document has no detectable heading structure at all.

3. **Extract** (`app/extraction.py`, `app/llm/prompts.py`) — sections are
   greedily packed into batches (multiple sections per prompt, each tagged
   by name, up to `MAX_BATCH_CHARS`) so one large document costs a handful
   of LLM calls rather than one per section — the single biggest lever on
   both speed and rate-limit exposure. Batches run with bounded concurrency
   (`MAX_CONCURRENT_LLM_CALLS`) against a round-robin pool of every
   configured Gemini/Groq key (`app/llm/pool.py`, `app/llm/factory.py`) —
   each additional key adds its own quota to the pool instead of sharing
   one. The prompt asks for a JSON array of facts with an open-ended
   `attribute` label the model invents per document, rather than a fixed
   enum — this is what lets the schema evolve as new kinds of documents
   come in. Each returned `verbatim_quote` is then re-located inside its
   section's source pages to pin down the exact page and character offset,
   so every fact keeps a hard link back to its evidence even though the
   LLM only saw a text blob, not page/offset coordinates.

4. **Normalize** (`app/normalization.py`) — plain Python, no LLM call: unit
   conversion (Cr/Lakh/Million/Billion, Indian comma grouping, parenthesized
   negatives), India's Apr–Mar fiscal-year parsing, date parsing via
   `dateutil`, and identifier cleanup (e.g. stripping a CIN's leading
   listing-status letter before comparing two CINs for identity).

5. **Compare** (`app/matching.py`, `app/comparison.py`) — for each new
   document's facts:
   - *Prefilter (code, no LLM):* find existing facts that share a
     normalized `entity_id`, overlap on `attribute` keywords, or fuzzy-match
     on `entity_name` (a normalized-similarity pass on the company/entity
     name itself). The fuzzy-name pass matters because exact-ID matching
     alone misses real corroborations when two documents word an entity
     differently (e.g. "Delhivery Limited" vs. "Delhivery Ltd.") — without
     it, that candidate pair would never even reach the LLM comparison
     step. Deliberately simple string matching, not embeddings — sufficient
     at the scale of a handful of documents (see Limitations).
   - *Comparison (LLM):* the prefiltered shortlist, plus the new document's
     own facts (so within-document consistency gets checked too), go into
     one digest-style prompt that judges each plausible pair as
     corroborating, contradicting, or reconciled by context — naming which
     dimension (time / scope / units / estimate vintage) explains an
     apparent conflict.

6. **Store, API, UI** — SQLite (`documents`, `facts`, `relationships`
   tables) only ever grows; new uploads are compared against the whole
   existing store and never re-derive anything already settled. FastAPI
   exposes `POST /documents` and `POST /documents/batch` (upload one or
   several PDFs, runs the full pipeline synchronously) and `GET`/
   `GET .../{id}` for both `facts` and `relationships`, each detail view
   returning the full evidence payload, plus `GET /stats` for running
   totals. A server-rendered UI (`/ui`, Jinja2, no frontend build step)
   gives a browser-based upload page (`/ui/upload`), a fact browser, a
   fact detail page (evidence quote front and center), and a relationship
   detail page showing both pieces of evidence side by side.

### Key decisions and trade-offs

- **No hard-coded schema.** `attribute` is free text the LLM invents per
  document; nothing is keyed to a specific document's field names, section
  titles, or entity list. This is what lets the same code run against PDFs
  it has never seen.
- **Deterministic where possible.** Normalization and prefiltering are
  plain code specifically so they're fast, free, testable in isolation,
  and don't add LLM-judgment risk to steps that don't need it.
- **Evidence is captured, not reconstructed.** Char offsets are recorded
  during parsing and re-located from the LLM's verbatim quote afterward,
  rather than trusting the LLM to report coordinates it was never given.
- **Fuzzy-name prefiltering over embeddings.** A lightweight
  `difflib`-based similarity pass on entity names (after stripping common
  corporate suffixes) closes the biggest gap in a pure-keyword prefilter
  without taking on an embedding index — the right amount of engineering
  for a handful of documents, with embeddings named explicitly as the next
  step if the fact count grows large (see Limitations).

### AI tools used

This project was built primarily with **Claude Code** (Anthropic's CLI
agent, running Claude Sonnet 5), using **parallel subagent-driven
development**: independent modules (PDF ingestion, the data/config/store
layer, the LLM client abstraction, normalization, section detection, the
matching prefilter, the extraction pipeline, the comparison pipeline, the
facts/relationships APIs, and the UI) were each built by a task-scoped
subagent working against an explicit interface contract, running
concurrently wherever modules had no file or data dependency on each
other, then integrated and verified against the full test suite after
each phase. Simpler, more mechanical modules (the read-only facts/
relationships list-and-detail APIs) were built by a smaller/cheaper model
(Haiku); modules with real design or correctness risk (parsing, matching,
extraction, comparison) used the larger model. Every module's generated
code was reviewed and test-verified (`pytest`) before being committed —
AI-generated code was not committed unreviewed.

## Limitations and Next Steps

- **Not yet run against the real starter dataset.** The pipeline is fully
  built and unit/integration-tested against synthetic fixtures and fake
  LLM clients, but hasn't yet produced a real run's output for the four
  required cases. **Next step before submission:** run it against the real
  starter dataset with real API keys, and swap in whatever the system
  actually surfaces for the four required cases in the demo video (see
  `docs/FOUR_REQUIRED_CASES.md`, kept out of this repo, for the
  hand-verified target examples used to validate the design while
  building).
- **A single free-tier key is genuinely rate-limited.** One Gemini
  free-tier key was observed capped at 5 requests/minute during
  development — the multi-key round-robin pool and section-batching (see
  Approach) are the two mitigations built for this, but if you're running
  with only one key and a large document, expect it to still take a
  while; add more keys (any mix of Gemini/Groq) to widen the pool.
- **Comparison doesn't scale past a handful of documents yet.** The
  comparison step sends a full digest of prefiltered candidates into one
  LLM call — fine at prototype scale (a few documents, a few hundred
  facts), but the digest would need to fit in one prompt at real "many
  PDFs" scale. Embedding-based candidate matching (replacing the string
  heuristics) is the natural next step if the fact count grows large.
- **Large-PDF performance is untested at scale.** Section-based chunking
  and concurrent extraction keep wall-clock time reasonable for
  moderate-length documents, but this hasn't been tested at 500+ pages.
- **The footnote-marker ambiguity (Case 4) is knowingly unresolved for
  2-decimal collisions.** A footnote digit glued onto a number with no
  separating space (e.g. "6.5⁴" read as "6.54") is indistinguishable by
  shape alone from a genuine two-decimal value. `normalize_quantity`
  strips a suspected footnote digit when there are 3+ digits after the
  decimal point (Indian filings essentially never report that much
  precision), but deliberately leaves the 2-decimal case alone rather than
  risk silently corrupting real values — documented and tested as a known
  limitation rather than force-fixed.
- **Sub-page evidence granularity.** A section's text is the full text of
  every page it spans; the evidence offset is precise to the character
  within a page, but a section that starts mid-page includes that whole
  page's text in the LLM prompt rather than a tighter sub-page slice.
- **Single-process, synchronous upload.** `POST /documents` runs the whole
  pipeline inline; fine for a demo-scale document, but a genuinely large
  PDF or a burst of concurrent uploads would benefit from a background-job
  queue with a status-poll endpoint instead.

## Additional Notes

- The `docs/` folder (planning notes, architecture rationale, and the
  hand-verified target examples for the four required cases) is kept out
  of this repository intentionally — it's build reference, not submission
  content — but informed every design decision described above.
- 138 automated tests cover every module (PDF parsing and table detection,
  sectioning, normalization, matching, batched extraction, comparison, the
  LLM pool/round-robin, and all API/UI routes), each using fixtures or
  fake LLM clients rather than live network calls, so the suite runs
  offline and deterministically.
