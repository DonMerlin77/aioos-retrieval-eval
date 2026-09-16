# AIOOS Retrieval Eval — Baseline Ledger

Frozen numbers to compare against when SuperSLM 1.5.0 + Dan's real embedder drop.
When the real embedder is in: change one line (`MODEL` in `eval.py`), rerun, add a new row below. Do not overwrite the stand-in row.

## How to reproduce (one command)
```
cd ~/aioos_retrieval_eval && python eval.py
```

## Provenance
- **script:** `eval.py`
- **input:** `memories.json` (12 memories, 6 queries, 1 positive-control probe)
- **instrument:** the embedder named in `eval.py:MODEL` + cosine-similarity KNN, k=3
- **controls:** positive control (near-copy must retrieve its exact memory at rank 1) + shuffle control (scrambled labels must collapse to ~0)

## Runs

### Run 1 — STAND-IN baseline (2026-09-16, graded-labels + CI upgrade)
- **embedder:** `sentence-transformers/all-MiniLM-L6-v2` (stand-in for Dan's Qwen3-0.6B-Embedding)
- **nDCG@3 = 0.84**  95% CI [0.63, 1.00]  ← DECIDING metric (graded gain, matches K1's convention)
- recall@3 = 0.92  95% CI [0.75, 1.00]
- hit@3 = 1.00  95% CI [1.00, 1.00]
- MRR = 0.92  95% CI [0.75, 1.00]
- positive control: PASS (near-copy retrieved `m6` at rank 1, expected `m6`)
- shuffle control: nDCG@3 = 0.00 (collapses from 0.84 → metric is real)
- note: labels are now graded (2=primary, 1=supporting) with a written rationale per pair in
  `memories.json`. The one real miss is q6: retrieves m9 (grade 1) but drops m12 (grade 2, the primary),
  which is why nDCG (0.84) sits below recall (0.92) — the grading catches a ranking error binary recall hides.
- CIs are WIDE because n=6. Do NOT read a two-model gap smaller than these bands as real; use
  `paired_diff_ci` (a CI straddling 0 = not resolvably different = report a tie). This is the number to beat/match.

### Run 1b — STAND-IN on the EXPANDED set (2026-09-16) — ADJUDICATED (committed baseline)
- **embedder:** `sentence-transformers/all-MiniLM-L6-v2`
- **set:** 40 memories (themed clusters), 30 queries across easy/medium/hard tiers, hard queries carry a hard_negative
- **nDCG@3 = 0.47**  95% CI [0.33, 0.61]  ← DECIDING. Much lower + tighter than Run 1's toy 0.84 — the bigger,
  harder corpus stopped inflating the score. THIS is the honest stand-in baseline to compare 1.5.0 against.
- recall@3 = 0.49 [0.34, 0.64]  ·  hit@3 = 0.63 [0.47, 0.80]  ·  MRR = 0.56 [0.43, 0.70]
- positive control PASS · shuffle control nDCG@3 = 0.06 (collapses) · hard-negative leakage 5/28
- **labels ADJUDICATED:** drafted by assistant → independently judge-audited (`audit_labels.py`, llama-3.3-70b) →
  Tom ruled on the disagreements. Changes: q9/m33 1→2, q11/m29→0, q13/m7 0→1, q16/m10 0→1 (hn m10→m19), q21/m38 0→1.
  Deciding metric moved 0.48→0.47 (adjudication was on the grade-1/hard-neg boundary, primaries were solid).
- audit trail: `audit_flags.json`. This IS the committed stand-in baseline; safe to compare 1.5.0 against.
- next lever for a tighter CI / stronger claim: grow queries toward ~40 (re-audit each addition the same way).

### Run 1c — STAND-IN, WIDENED set (2026-09-16) — ADJUDICATED (current committed baseline)
- **embedder:** `sentence-transformers/all-MiniLM-L6-v2`
- **set:** 60 memories, 55 queries (easy/medium/hard tiers), hard queries carry a hard_negative
- **nDCG@3 = 0.53**  95% CI [0.43, 0.64]  ← DECIDING. CI half-width tightened from ±0.14 (n=30) to ±0.10.
- recall@3 = 0.53 [0.43, 0.64] · hit@3 = 0.73 [0.60, 0.84] · MRR = 0.63 [0.53, 0.73]
- positive control PASS · shuffle control nDCG@3 = 0.09 (collapses) · hard-negative leakage 8/53
- labels ADJUDICATED (drafted → independent-judge audit `audit_labels.py` → Tom ruled). See memories.json `_status`.
- **THIS is the current committed stand-in baseline; compare 1.5.0 against it via `paired_diff_ci` on per-query nDCG@3.**
- (Run 1b n=30 nDCG@3=0.47 was the intermediate stage before widening; kept above for the history.)

### Run 2 — REAL embedder (SuperSLM 1.5.0 + Qwen3-0.6B-Embedding) — PENDING
- run with `AIOOS_EMBEDDER=<his-model> python eval.py`, paste numbers here (no file edit)
- read to report to Dan: does int8 hurt retrieval vs float, and does the real embedder hold vs the stand-in
- the comparison is only honest through `paired_diff_ci(stand_in_perq, real_perq)` on per-query nDCG@3:
  if that CI straddles 0, they are NOT resolvably different — report a tie, do not claim a winner
- both controls (positive + shuffle) must still pass on the real embedder or the comparison is void
