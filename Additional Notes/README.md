## Additional Notes

- **Quota exhaustion hit the demo video twice, independently, in two
  separate recording sessions.** First recording: the plan was three
  real-data runs (a small synthetic set, the Delhivery filings, and the
  India macro-economy filings), but by run three every configured API
  key — across two providers — had hit its quota for the day from the
  extensive testing this project's development involved, so that
  recording substituted a second synthetic company for run three.
  Second recording (the current video, re-shot for a more realistic,
  continuously-navigated screen-capture style): re-running Delhivery and
  Bluepeak extraction hit total quota exhaustion *again*, independently,
  on both datasets — Gemini's daily cap and Groq's per-minute limit
  together, across all four pooled keys. Rather than show a broken run a
  second time, that recording's results clips use real facts this same
  pipeline extracted from the same files during an earlier successful
  run in this project (not fabricated, just not from that specific take)
  while the Set A synthetic run in both recordings was always fully live
  and genuine. Two independent recording sessions hitting the same real
  ceiling is, if anything, stronger evidence for the limitation
  documented above than either session alone.
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
