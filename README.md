# AIOOS Retrieval Eval

A small, honest benchmark for one question: **when a game NPC is in a situation, does the embedder retrieve the memories it should actually recall to respond well?**

This is an *application-facing* retrieval eval, not an academic IR benchmark. Academic benchmarks (BEIR and friends) ask "is this embedder good at ranking scientific abstracts." That is a different question from "does the tavern keeper remember the washed-out bridge when a traveler asks about the north road." This measures the second one, because that is what a living-NPC system actually needs.

## What it measures

An NPC (a tavern keeper) has a set of memories. Each query is a situation. Graded labels say which memories *should* come back, and why.

- **nDCG@3** is the deciding metric (graded relevance, log2 discount, the standard TREC convention). It rewards putting the must-recall memory above a merely-related one.
- **recall@3 / hit@3 / MRR** are reported as diagnostics, never as the headline.
- Every number is reported with a **bootstrap 95% CI** over the queries, so a small-sample score is never mistaken for a precise one.

## The labels are the ground truth, so they are treated like it

Retrieval evals live or die on their relevance labels. Here every labeled pair carries a **grade** (2 = primary/must-recall, 1 = supporting) and a **written rationale**, so the benchmark documents its own definition of "correct" instead of resting on one person's unstated judgment. See `memories.json`.

Labels were built by a deliberately separated process:
1. drafted,
2. re-graded by an **independent judge** (a different model, `audit_labels.py`) under the same rubric,
3. every disagreement **adjudicated by a human**, whose ruling is the ground truth of record.

The generator of a label is never its sole grader. That separation is the whole point.

## Controls (why you can trust a number here)

Each control is a cheap test that would *fail* if the benchmark were fake:

- **Positive control** — a near-copy query must retrieve its exact memory at rank 1, or the pipeline is broken.
- **Shuffle control** — scramble which labels belong to which query and the score must collapse to ~0, or the metric is measuring nothing.
- **Hard-negative control** — most queries name a topically-close memory that should *not* be retrieved (e.g. "the north road is safe" as a trap against "the bridge is washed out"). If these keep landing in the top-k, the embedder is topic-matching, not judging relevance. Leakage is reported by name.
- **Independent-judge audit** — the label audit above.

## An honest baseline (and why the number went down)

Same stand-in embedder (`all-MiniLM-L6-v2`), two versions of the benchmark:

| Set | nDCG@3 | 95% CI |
|---|---|---|
| toy (12 memories, 6 queries) | 0.84 | [0.63, 1.00] |
| realistic (60 memories, 55 queries, hard negatives) | **0.53** | [0.43, 0.64] |

The score *dropped* on the bigger set. That is the benchmark working: a tiny, easy corpus inflates retrieval scores because there is nothing to confuse the embedder. The realistic set with topical hard negatives is where retrieval quality actually gets tested. A benchmark that always reports high numbers is not measuring anything.

## Run it

```sh
pip install -r requirements.txt
python eval.py
```

Point it at a different embedder with no code change:

```sh
AIOOS_EMBEDDER=<any sentence-transformers model> python eval.py
```

Comparing two embedders honestly: run each through `per_query_scores(...)['nDCG@3']` and pass both to `paired_diff_ci`. If the confidence interval on the per-query difference straddles 0, the two are **not resolvably different** on this benchmark, report a tie, not a winner. Small differences inside the noise are not results.

## Files

- `eval.py` — the benchmark: embed, rank, score, controls, CIs.
- `memories.json` — the corpus, queries, graded+rationaled labels, hard negatives.
- `audit_labels.py` — the independent-judge label audit (needs an `OPENROUTER_API_KEY`).
- `BASELINE.md` — the frozen baseline ledger, with provenance and a slot for each new run.

## Status

Stand-in embedder for now. Built to swap in a production embedder and produce a clean before/after read (including whether int8 quantization costs retrieval quality) with the controls still holding.

## License

MIT.
