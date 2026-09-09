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
- **Groq also rejects oversized individual requests outright (`413 Payload
  Too Large`), separately from rate-limiting.** Found while re-running
  extraction on the real Delhivery filings (227-page prospectus): a
  document's batched extraction call can fail this way even with fresh,
  unthrottled keys and no concurrent contention on them, because the
  failure is about a single request's size, not the account's quota. Same
  degrade-to-zero-facts behavior as quota exhaustion, and the same
  work-around (a fresh solo upload of just that document, letting
  section-batching produce smaller individual calls, recovers it most of
  the time). Next step: shrink `MAX_BATCH_CHARS` when a batch's target
  provider is Groq specifically, rather than using one constant for every
  provider in the pool.
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
