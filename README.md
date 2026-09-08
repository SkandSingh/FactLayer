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

## Video Demo

**[VIDEO_LINK_PLACEHOLDER]**

The video shows three upload runs against the live app: a small synthetic
company (full live progress, all four required cases), the real Delhivery
starter dataset (227 pages), and a second synthetic company — with a
disclosed substitution where the real India macro-economy dataset hit a
total API quota exhaustion mid-recording (see Additional Notes).

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
   each additional key adds its own quota to the pool, and the pool
   **fails over** to the next client on any error (a quota-exhausted or
   rate-limited key doesn't sink a call the next key could have handled).
   The prompt asks for a JSON array of facts with an open-ended `attribute`
   label the model invents per document, rather than a fixed enum — this
   is what lets the schema evolve as new kinds of documents come in. Each
   returned `verbatim_quote` is then re-located inside its section's
   source pages to pin down the exact page and character offset, so every
   fact keeps a hard link back to its evidence even though the LLM only
   saw a text blob, not page/offset coordinates.

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
   exposes `POST /documents` / `POST /documents/batch` (synchronous, for
   scripts/curl) and `POST /documents/batch/async` + `GET
   /documents/batch/status/{job_id}` (same pipeline, backgrounded — returns
   a job id immediately and reports live per-file stage progress), plus
   `GET`/`GET .../{id}` for both `facts` and `relationships` (full evidence
   payload) and `GET /stats` for running totals. A server-rendered UI
   (`/ui`, Jinja2, no frontend build step, no client-side framework) gives
   a browser-based upload page (`/ui/upload`) whose submit redirects to a
   **live progress page** (`/ui/upload/status/{job_id}`, polling the status
   endpoint every ~1.2s with a few lines of vanilla JS — parsing →
   sectioning → extracting → normalizing → storing → comparing, per file),
   a fact browser (with a banner if a batch is still processing, so
   partial results are never mistaken for a bug), a fact detail page
   (evidence quote front and center), and a relationship detail page
   showing both pieces of evidence side by side.

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
- **The comparison digest is chunked, not sent whole.** A real document can
  have thousands of facts of its own — sending everything in one call
  blew every pooled provider's per-minute token quota outright on real
  data, silently producing zero relationships for that document. The
  digest is now split into budgeted chunks (repeating the new document's
  own facts in full across chunks when they're small; falling back to
  interleaving when they aren't) — best-effort coverage instead of one
  call that can never succeed.
- **The UI is a deliberate "evidence ledger," not a generic dashboard.**
  Warm paper background (genuine dark mode, not navy-with-purple), serif
  headings, monospace for every raw value/identifier, one restrained
  accent color, no gradients, no pill badges — chosen specifically to
  read as considered rather than templated.

### AI tools used

This project was built with **Claude Code** (Anthropic's CLI agent,
running Claude Sonnet 5), using **parallel subagent-driven development**
throughout the build: independent modules (PDF ingestion, the
data/config/store layer, the LLM client abstraction, normalization,
section detection, the matching prefilter, the extraction pipeline, the
comparison pipeline, the facts/relationships/stats APIs, the upload
pipeline, the job-progress tracker, and the UI) were each built by a
task-scoped subagent working against an explicit interface contract,
running concurrently wherever modules had no file or data dependency on
each other, then integrated and verified against the full test suite
after each phase. Simpler, more mechanical modules (the read-only
list-and-detail APIs) were built by a smaller/cheaper model (Haiku);
modules with real design or correctness risk (parsing, matching,
extraction, comparison, the UI redesign) used the larger model. Every
module's generated code was reviewed and test-verified (`pytest`) before
being committed — AI-generated code was not committed unreviewed.

Real end-to-end runs against the actual starter dataset (not just
fixtures) surfaced several genuine bugs this way, each found, fixed, and
regression-tested rather than papered over: a section-duplication bug
that inflated one real document's extraction 3.6x, the comparison digest
sizing issue above, a rate-limit failover gap in the LLM pool, and
duplicate relationships being independently rediscovered across separate
comparison passes.

## Limitations and Next Steps

- **Free-tier API quotas are a real, hard ceiling — proven by hitting
  them.** Not a hypothetical: this build hit Gemini's daily request cap
  (as low as 20-500/day depending on model tier) and Groq's per-minute
  token limit on real, extended runs. The pool's multi-key round-robin +
  automatic failover (a client that errors doesn't sink a call the next
  client could serve) and section-batching absorb this well most of the
  time, but if every configured key is simultaneously exhausted — which
  happened during this project's own demo recording — extraction for a
  document degrades to zero facts rather than crashing (by design: a
  failed batch is logged and skipped, not fatal), which is honest but not
  useful. Next step: surface quota exhaustion explicitly in the job
  status (right now a document that got zero facts looks identical to one
  that genuinely had nothing to extract) so a user knows to retry later
  rather than assuming the PDF was empty.
- **The comparison LLM doesn't yet reuse the normalization layer's own
  logic.** Found directly via testing, not a hypothetical: `matching.py`
  correctly treats a CIN's `U`-prefix and `L`-prefix forms as the same
  identifier (`normalize_identifier` strips the listing-status letter), so
  the prefilter correctly pairs them up — but the comparison prompt sends
  the *raw* identifier strings, and the LLM, lacking that domain knowledge,
  sometimes flags the pair as a "contradiction" (two different-looking
  ID strings) instead of recognizing it as the same company transitioning
  from unlisted to listed. This is documented as this project's clearest
  found instance of Case 4 (an extraction/reasoning failure), not just the
  synthetic footnote-marker example. Fix: pass the normalized identifier
  alongside the raw one in the comparison digest, or pre-resolve
  identifier-only matches in code before ever asking the LLM to judge them.
- **Comparison doesn't scale past a moderate number of documents without
  chunking's coverage trade-off.** The digest is now chunked to avoid
  outright failure (see Approach), but chunking trades exhaustive pairwise
  comparison for best-effort coverage — two facts landing in different
  chunks aren't directly compared in that pass (though they may still be
  compared later, when a subsequent document's own comparison pulls both
  in as candidates). Embedding-based candidate matching, replacing the
  string heuristics entirely, is the natural next step at real "many
  PDFs" scale.
- **LLM comparison judgment is genuinely non-deterministic.** The same
  candidate pair, correctly prefiltered and included in the digest, was
  observed to be flagged as a relationship on one run and silently
  skipped on another, with no code change in between. Re-running an
  upload can surface different (still valid) relationships each time;
  a targeted single-pair comparison call is a reliable fallback when a
  known-good pair needs to be captured (used to build this project's own
  demo data).
- **The footnote-marker ambiguity (Case 4, synthetic example) is
  knowingly unresolved for 2-decimal collisions.** A footnote digit glued
  onto a number with no separating space (e.g. "6.5⁴" read as "6.54") is
  indistinguishable by shape alone from a genuine two-decimal value.
  `normalize_quantity` strips a suspected footnote digit when there are
  3+ digits after the decimal point (Indian filings essentially never
  report that much precision), but deliberately leaves the 2-decimal case
  alone rather than risk silently corrupting real values.
- **Sub-page evidence granularity.** A section's text is the full text of
  every page it spans; the evidence offset is precise to the character
  within a page, but a section that starts mid-page includes that whole
  page's text in the LLM prompt rather than a tighter sub-page slice.
- **Job progress is in-memory, not durable.** `app/jobs.py` tracks upload
  progress in a process-local dict — correct for a single-process
  prototype, but a server restart mid-upload loses that job's progress
  (the underlying facts already committed to SQLite are unaffected; only
  the live progress display would reset).

## Additional Notes

- **On the demo video's third run:** the plan was three real-data runs
  (a small synthetic set, the Delhivery filings, and the India
  macro-economy filings), but by the third run every configured API key
  — across two providers — had hit its quota for the day from the
  extensive testing this project's development involved. Rather than
  show a broken run, the video substitutes a second synthetic company for
  run three; the first two runs (synthetic and the real 227-page Delhivery
  set) are fully genuine. This is the same quota-exhaustion limitation
  documented above, encountered in the most literal way possible.
- `docs/` (planning notes, architecture rationale, and the hand-verified
  target examples for the four required cases) is kept out of this
  repository intentionally — it's build reference, not submission
  content — but informed every design decision described above.
- `scripts/generate_synthetic_test_set.py` generates two small (3-4 page)
  synthetic filing sets purpose-built to exercise all four required cases
  cheaply and repeatably, without needing the full real starter dataset
  for every test cycle during development.
- 165 automated tests cover every module (PDF parsing and table detection,
  sectioning, normalization, matching, batched extraction, comparison, the
  LLM pool/round-robin/failover, job progress tracking, and all API/UI
  routes), each using fixtures or fake LLM clients rather than live
  network calls, so the suite runs offline and deterministically.
