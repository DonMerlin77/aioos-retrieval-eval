"""
AIOOS retrieval-quality eval. Measures whether the embedder + nearest-neighbor search returns the
SEMANTICALLY RIGHT memories for a situation. Stand-in embedder now (all-MiniLM); swap in Dan's later
via AIOOS_EMBEDDER=... (see BASELINE.md). This is the APPLICATION-facing complement to Dan's K1 gate:
K1 asks "is the embedder good on BEIR benchmarks," this asks "does it recall the right memory for a
game moment." Labels are graded (2=primary, 1=supporting) with a written rationale per pair in
memories.json, so the eval documents its own definition of correct.

Metrics:
  nDCG@k    : the DECIDING metric (graded gain, log2 discount) -- matches K1's convention so the two
              evals speak the same language
  recall@k  : of the memories that SHOULD come back, how many landed in the top k (binary, grade>0)
  hit@k     : did at least one right memory land in the top k
  MRR       : how high the first right memory ranked over the full ranking (1.0 = always rank 1)
Honesty (the part that makes a number trustworthy):
  bootstrap 95% CI : every headline metric is reported with an error bar from resampling the queries,
                     so a small-n number (n=6 today) is never mistaken for a precise one
  positive control : a near-copy query MUST retrieve its exact memory at rank 1, or the eval is broken
  shuffle control  : with the labels scrambled, the score MUST collapse to ~chance, or the metric is fake
  paired_diff_ci   : the NOISE FLOOR for comparing two embedders -- 95% CI on the per-query nDCG
                     difference; if it straddles 0 the two models are not resolvably different (this is
                     what makes a stand-in-vs-real comparison honest instead of anecdotal)
"""
import json, math, os, random
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
# stand-in default; point at Dan's embedder with AIOOS_EMBEDDER=... (no file edit needed). See BASELINE.md.
MODEL = os.getenv("AIOOS_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
K = 3
N_BOOT = 10000
SEED = 0


def load():
    d = json.load(open(HERE / "memories.json"))
    return d["memories"], d["queries"], d["positive_control"]


def grades(q):
    """{mem_id: grade} for one query, from the graded labels."""
    return {mid: lab["grade"] for mid, lab in q["labels"].items()}


def embed(model, texts):
    v = model.encode(texts, normalize_embeddings=True)   # unit vectors -> dot product == cosine similarity
    return np.asarray(v)


def rank_memories(qvec, mvecs, mem_ids):
    sims = mvecs @ qvec                      # cosine similarity to every memory
    order = np.argsort(-sims)                # best first
    return [mem_ids[i] for i in order]


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def per_query_scores(queries, ranked_by_q, k=K):
    """Returns per-query dicts (keyed by query id) for each metric, so any of them can be bootstrapped
    or paired against another model's run."""
    ndcg, recall, hit, rr = {}, {}, {}, {}
    for q in queries:
        g = grades(q)
        rel = {mid for mid, grade in g.items() if grade > 0}
        ranked = ranked_by_q[q["id"]]
        topk = ranked[:k]

        ideal = sorted((v for v in g.values() if v > 0), reverse=True)[:k]
        idcg = _dcg(ideal)
        gains = [g.get(mid, 0) for mid in topk]
        ndcg[q["id"]] = (_dcg(gains) / idcg) if idcg > 0 else 0.0

        found = rel & set(topk)
        recall[q["id"]] = len(found) / len(rel) if rel else 0.0
        hit[q["id"]] = 1.0 if found else 0.0
        first = next((i for i, mid in enumerate(ranked, 1) if mid in rel), None)
        rr[q["id"]] = 1.0 / first if first else 0.0
    return {"nDCG@%d" % k: ndcg, "recall@%d" % k: recall, "hit@%d" % k: hit, "MRR": rr}


def ndcg_only(queries, ranked_by_q, k):
    """Per-query nDCG@k for an arbitrary k (so we can report @3/@5/@10 and show k=3 was not
    cherry-picked)."""
    out = {}
    for q in queries:
        g = grades(q)
        ideal = sorted((v for v in g.values() if v > 0), reverse=True)[:k]
        idcg = _dcg(ideal)
        gains = [g.get(mid, 0) for mid in ranked_by_q[q["id"]][:k]]
        out[q["id"]] = (_dcg(gains) / idcg) if idcg > 0 else 0.0
    return out


def executed_chance(queries, mem_ids, k=K, trials=1000, seed=SEED):
    """The true chance floor: mean nDCG@k of uniformly RANDOM rankings (design mirrors K1's
    executed_chance_level). Complements the shuffle control -- shuffle scrambles the labels,
    this randomizes the ranking; a real metric must sit well above both."""
    rng = np.random.default_rng(seed)
    per_trial = []
    for _ in range(trials):
        rand = {q["id"]: list(rng.permutation(mem_ids)) for q in queries}
        per_trial.append(float(np.mean(list(ndcg_only(queries, rand, k).values()))))
    return float(np.mean(per_trial)), float(np.std(per_trial))


def bootstrap_ci(per_query, n_boot=N_BOOT, seed=SEED):
    """95% CI on the mean, by resampling the queries with replacement. With n=6 this CI is wide on
    purpose -- it is the honest statement that a 6-query mean is not a precise number."""
    vals = np.array(list(per_query.values()), dtype=np.float64)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(per_query_a, per_query_b, n_boot=N_BOOT, seed=SEED):
    """The noise floor for a two-model comparison: 95% CI on the per-query difference a - b (same
    queries, same labels, two rankings). If the CI straddles 0, the two embedders are NOT resolvably
    different on this eval -- report them as tied, do not claim a winner. This is the guard that keeps
    a stand-in-vs-real read from turning a 0.92-vs-0.94 into a fake win. Mirrors K1's paired resolving
    power in spirit (a paired CI, not either model's marginal spread)."""
    shared = sorted(set(per_query_a) & set(per_query_b))
    diffs = np.array([per_query_a[q] - per_query_b[q] for q in shared], dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), n


def run_model(model, memories, queries):
    """Embed + rank once; returns {query_id: full_ranked_mem_ids} for this model."""
    mem_ids = [m["id"] for m in memories]
    mvecs = embed(model, [m["text"] for m in memories])
    qvecs = embed(model, [q["text"] for q in queries])
    return {q["id"]: rank_memories(qvecs[i], mvecs, mem_ids) for i, q in enumerate(queries)}, mvecs, mem_ids


GOAL = (
    "GOAL: measure whether the embedder retrieves the memories an NPC should actually recall to\n"
    "respond well in a given situation. This is the AIOOS application target, not academic IR.\n"
    "A high score here means: when a game moment calls for a memory, the right one comes back."
)

LEGEND = {
    "nDCG@%d" % K: "DECIDING metric. Rewards putting the most-relevant memory (grade 2) above a\n"
                   "               merely-supporting one (grade 1), in the top %d. Same convention as Dan's K1\n"
                   "               gate, so the two evals compare like-for-like. This is the number to trust." % K,
    "recall@%d" % K: "diagnostic. Of the memories that SHOULD come back, how many made the top %d." % K,
    "hit@%d" % K: "diagnostic. Did at least one right memory make the top %d (did we recall ANYTHING useful)." % K,
    "MRR": "diagnostic. How high the first right memory ranked (1.0 = always first). Speed-to-relevance.",
}


def main():
    random.seed(0)
    memories, queries, pc = load()
    print(f"loading embedder: {MODEL}")
    model = SentenceTransformer(MODEL)

    ranked, mvecs, mem_ids = run_model(model, memories, queries)
    scores = per_query_scores(queries, ranked)

    print("\n" + GOAL)
    print(f"\n== RETRIEVAL QUALITY (n={len(queries)} queries, {len(memories)} memories, k={K}) ==")
    print("  each metric is reported with a 95% bootstrap CI over queries; a wide band = small-n, not precise.\n")
    for name, per_q in scores.items():
        mean, lo, hi = bootstrap_ci(per_q)
        print(f"  {name:<10} = {mean:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"     why: {LEGEND[name]}")

    # SENSITIVITY: the smallest quality difference this eval can reliably DETECT at this n.
    # Reported as the deciding-metric CI half-width -- a conservative bound (paired comparisons of
    # similar embedders resolve finer, since the same queries correlate). A regression smaller than
    # this is not reliably detectable at the current n; grow the query set to see smaller ones.
    nd_mean, nd_lo, nd_hi = bootstrap_ci(scores["nDCG@%d" % K])
    sensitivity = (nd_hi - nd_lo) / 2
    print(f"\n  eval sensitivity: resolves nDCG@{K} differences of about >= {sensitivity:.2f} at n={len(queries)} "
          f"(conservative; paired int8-vs-float does better).")
    print(f"     why: this is the eval's usefulness -- the smallest embedder regression it can reliably catch.")

    # nDCG at other k, to show k=3 was not cherry-picked
    print("\n  nDCG at other cutoffs (k=3 is the headline; these show it is not cherry-picked):")
    for kk in (5, 10):
        m, lo, hi = bootstrap_ci(ndcg_only(queries, ranked, kk))
        print(f"    nDCG@{kk:<2} = {m:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")

    # EXECUTED CHANCE FLOOR: mean nDCG@3 of random rankings (the true floor)
    ch_mean, ch_std = executed_chance(queries, mem_ids, k=K)
    print(f"\n  executed chance floor: random-ranking nDCG@{K} = {ch_mean:.3f} (std {ch_std:.3f}, 1000 trials)")
    print(f"     why: the real score ({np.mean(list(scores['nDCG@%d' % K].values())):.2f}) must sit far above this floor, or it is not retrieval.")

    # POSITIVE CONTROL: a near-copy query must retrieve its exact memory at rank 1
    pcvec = embed(model, [pc["text"]])[0]
    pc_top1 = rank_memories(pcvec, mvecs, mem_ids)[0]
    ok = pc_top1 == pc["must_retrieve"]
    print(f"\n  positive control: near-copy retrieved '{pc_top1}' at rank 1, expected '{pc['must_retrieve']}'  ->  {'PASS' if ok else 'FAIL (eval is broken)'}")

    # HARD-NEGATIVE CONTROL: each query names a topically-close memory that should NOT be
    # retrieved. If these keep landing in the top-k, the embedder is topic-matching, not judging
    # relevance -- so the score would be flattered by surface similarity. Lower leakage = better.
    hn_queries = [q for q in queries if q.get("hard_negative")]
    leaked = [q for q in hn_queries if q["hard_negative"] in ranked[q["id"]][:K]]
    print(f"\n  hard-negative leakage: {len(leaked)}/{len(hn_queries)} queries retrieved their tempting"
          f" NON-relevant memory into top{K}")
    print(f"     why: proves the metric measures RELEVANCE, not topic overlap. High leakage = the score is fake-ish.")
    if leaked:
        print("     leaked on: " + ", ".join(f"{q['id']}(->{q['hard_negative']})" for q in leaked))

    # SHUFFLE CONTROL: scramble which label-set belongs to which query; nDCG MUST collapse
    label_sets = [q["labels"] for q in queries]
    shuffled = label_sets[:]; random.shuffle(shuffled)
    shuf_queries = [{"id": q["id"], "labels": shuffled[i]} for i, q in enumerate(queries)]
    shuf_ndcg = per_query_scores(shuf_queries, ranked)["nDCG@%d" % K]
    smean, _, _ = bootstrap_ci(shuf_ndcg)
    real_ndcg_mean = np.mean(list(scores["nDCG@%d" % K].values()))
    print(f"  shuffle control : nDCG@{K} with scrambled labels = {smean:.2f}  (should be << {real_ndcg_mean:.2f}; if not, the metric is fake)")

    print("\n  per-query (what came back vs what should have):")
    for q in queries:
        g = grades(q)
        want = ", ".join(f"{mid}={grd}" for mid, grd in g.items())
        print(f"    {q['id']}: want [{want}]  got top{K} {ranked[q['id']][:K]}")

    print("\n  (to compare two embedders honestly, run each into per_query_scores(...)['nDCG@%d'] and\n   pass both to paired_diff_ci -- a CI straddling 0 means 'not resolvably different', report a tie.)" % K)


if __name__ == "__main__":
    main()
