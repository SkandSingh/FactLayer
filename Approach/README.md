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
