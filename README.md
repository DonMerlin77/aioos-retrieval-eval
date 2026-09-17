# AIOOS Retrieval Eval

A small, honest benchmark for one question: **when a game NPC is in a situation, does the embedder retrieve the memories it should actually recall to respond well?**

This is an *application-facing* retrieval eval, not an academic IR benchmark. Academic benchmarks (BEIR and friends) ask "is this embedder good at ranking scientific abstracts." That is a different question from "does the tavern keeper remember the washed-out bridge when a traveler asks about the north road." This measures the second one, because that is what a living-NPC system actually needs.

## What it measures

An NPC (a tavern keeper) has a set of memories. Each query is a situation. Graded labels say which memories *should* come back, and why.

- **nDCG@3** is the deciding metric (graded relevance, log2 discount, the standard TREC convention). It rewards putting the must-recall memory above a merely-related one.
- **recall@3 / hit@3 / MRR** are reported as diagnostics, never as the headline.
- Every number is reported with a **bootstrap 95% CI** over the queries, so a small-sample score is never mistaken for a precise one.

## How this differs from an intrinsic embedder benchmark (BEIR, MTEB, and gates built on them)

Intrinsic benchmarks measure whether an embedder is good *in general*, on formal corpora (scientific abstracts, arguments, finance), with relevance defined by topical/semantic match. That is a real and necessary question, and it is a different question from this one. This benchmark is *task-situated*: it measures whether an embedder is good *for a living-NPC memory system*, and it differs on three axes an intrinsic benchmark cannot cover:

1. **A different relevance construct.** Here "relevant" means *would recalling this help the NPC act well*, which includes behavioral relevance an intrinsic benchmark has no notion of. Example: to a mother asking if it is safe for her kids to play outside, a memory about a collapsed mine outside town is relevant as a *warning*, even though it is not the topical answer to "safe." The labels encode that.
2. **A different data distribution.** Short, first-person, episodic memories in a narrative domain, not formal documents. An embedder can score well on intrinsic IR and worse on short narrative memory (or the reverse), so this catches regressions an intrinsic benchmark structurally cannot see.
3. **Application-specific distractors.** The hard negatives are built from in-world confusability: contradicting memories, wrong-direction debts, right-topic-wrong-time. An intrinsic benchmark never tests those.

The two are complementary: an intrinsic benchmark gates the embedder itself; this validates the embedder *in the application*. You can pass one and fail the other.

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
- **Executed chance floor** — the mean nDCG of 1000 uniformly random rankings (~0.04). The real score must sit far above it, or it is not retrieval. (This is a stronger floor than shuffle: shuffle scrambles the labels, this randomizes the ranking.)
- **Hard-negative control** — most queries name a topically-close memory that should *not* be retrieved (e.g. "the north road is safe" as a trap against "the bridge is washed out"). If these keep landing in the top-k, the embedder is topic-matching, not judging relevance. Leakage is reported by name.
- **Independent-judge audit (two passes)** — (1) every labeled pair is re-graded by a separate model and disagreements are human-adjudicated; (2) a *missed-label* audit re-grades the memories the embedder retrieved but the labels never credited, so an incomplete label set cannot silently penalize a good embedder. See `audit_labels.py` and `audit_missed.py`.

nDCG is also reported at k=5 and k=10, so the headline k=3 is visibly not cherry-picked.

## An honest baseline (and why the number went down)

Same stand-in embedder (`all-MiniLM-L6-v2`), two versions of the benchmark:

| Set | nDCG@3 | 95% CI |
|---|---|---|
| toy (12 memories, 6 queries) | 0.84 | [0.63, 1.00] |
| realistic (60 memories, 55 queries, hard negatives) | **0.53** | [0.43, 0.64] |

The score *dropped* on the bigger set. That is the benchmark working: a tiny, easy corpus inflates retrieval scores because there is nothing to confuse the embedder. The realistic set with topical hard negatives is where retrieval quality actually gets tested. A benchmark that always reports high numbers is not measuring anything.

## Does the eval have discriminating power? (why it's worth running)

An eval only matters if it can tell good retrieval from bad. `compare_methods.py` runs three
strategies through the same benchmark:

| method | nDCG@3 | 95% CI |
|---|---|---|
| embedder (semantic) | 0.55 | [0.45, 0.65] |
| lexical (word overlap, no embeddings) | 0.41 | [0.30, 0.52] |
| random | 0.03 | [0.01, 0.06] |

All three gaps are **resolvable** (the paired CI on the per-query difference excludes 0). The
embedder resolvably beats keyword matching (+0.14 [+0.03, +0.24]), the analogue of an intrinsic
gate's "beat BM25" floor. And the per-query breakdown shows *what breaks without good retrieval*:
asked "is anyone a threat to me," keyword matching returns "a family arrived from a burned village"
while the embedder returns the threatening note. The eval separates the methods and names the
failures, which is the whole reason to run it.

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

## Scope and limits (what it does and does not claim)

- It measures **retrieval quality** (does the right memory come back), not end-to-end NPC response quality. Retrieval is a necessary input to a good response, not the whole of it.
- The corpus is a **single NPC archetype** (a tavern keeper) and is **synthetic and single-author**, so results describe *this benchmark*, not game retrieval in general. Broadening to more NPC types and domains is the path to a wider claim.
- Ground-truth relevance is **human-adjudicated after an independent-model audit**, both for the labels present and for labels that might be missing. It is a considered judgment of what an NPC should recall, not an absolute.
- Two embedders are only called different when the **paired confidence interval on their per-query difference excludes 0**. Differences inside the noise are reported as ties.

These are stated plainly because a benchmark that hides its limits cannot be trusted about anything else.

## Status

Stand-in embedder for now. Built to swap in a production embedder and produce a clean before/after read (including whether int8 quantization costs retrieval quality) with the controls still holding.

## License

MIT.
