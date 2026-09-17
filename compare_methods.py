"""
Does the eval actually distinguish good retrieval from bad? (The value-of-the-eval test.)

An eval only matters if it separates a real method from a naive one. This runs three retrieval
strategies through the SAME benchmark and asks whether the eval resolvably ranks them:

  embedder : the real thing (semantic nearest-neighbor)
  lexical  : word-overlap ranking -- "what you get from keyword matching, no embeddings" (a floor,
             the analogue of K1's BM25 F2 floor)
  random   : shuffled ranking -- the null

If the eval is worth anything: embedder > lexical > random, each gap outside the paired-difference
CI (i.e. resolvable, not noise). It also answers "what happens without good retrieval" concretely:
the per-query breakdown shows situations where the naive method grabs the WRONG memory that the
embedder gets right -- exactly the NPC failure the eval exists to catch before it ships.
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from eval import load, run_model, per_query_scores, bootstrap_ci, paired_diff_ci, grades, MODEL, K

STOP = set(("a an the of to in on for and or but is are was were be been am i my me you your he she "
            "it they them his her their we our this that these those with as at by from about into "
            "whether any some no not do does did have has had will would can could should there here "
            "what who when where why how me my mine if then than so out up over").split())


def toks(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}


def lexical_rank(queries, memories):
    """Rank memories by word overlap with the query (Jaccard), deterministic id tie-break."""
    mtoks = {m["id"]: toks(m["text"]) for m in memories}
    ranked = {}
    for q in queries:
        qt = toks(q["text"])
        scored = []
        for m in memories:
            inter = len(qt & mtoks[m["id"]])
            union = len(qt | mtoks[m["id"]]) or 1
            scored.append((m["id"], inter / union))
        scored.sort(key=lambda x: (-x[1], x[0]))
        ranked[q["id"]] = [mid for mid, _ in scored]
    return ranked


def random_rank(queries, mem_ids, seed=0):
    rng = np.random.default_rng(seed)
    return {q["id"]: list(rng.permutation(mem_ids)) for q in queries}


def ndcg_perq(queries, ranked):
    return per_query_scores(queries, ranked)["nDCG@%d" % K]


def main():
    memories, queries, _ = load()
    mem_ids = [m["id"] for m in memories]
    mem_text = {m["id"]: m["text"] for m in memories}
    print(f"loading embedder: {MODEL}")
    model = SentenceTransformer(MODEL)

    emb_ranked, _, _ = run_model(model, memories, queries)
    lex_ranked = lexical_rank(queries, memories)
    rnd_ranked = random_rank(queries, mem_ids)

    runs = {"embedder": emb_ranked, "lexical": lex_ranked, "random": rnd_ranked}
    perq = {name: ndcg_perq(queries, r) for name, r in runs.items()}

    print(f"\n== nDCG@{K} by retrieval method (n={len(queries)}) ==")
    for name in ("embedder", "lexical", "random"):
        m, lo, hi = bootstrap_ci(perq[name])
        print(f"  {name:<9} = {m:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n== resolvable? (paired 95% CI on the per-query difference; excludes 0 = real gap) ==")
    for a, b in (("embedder", "lexical"), ("embedder", "random"), ("lexical", "random")):
        mean, lo, hi, n = paired_diff_ci(perq[a], perq[b])
        verdict = "RESOLVABLE (real gap)" if (lo > 0 or hi < 0) else "tie (within noise)"
        print(f"  {a} - {b}: {mean:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  ->  {verdict}")

    # What breaks WITHOUT good retrieval: queries the embedder gets but lexical fails.
    print("\n== what happens without the embedder (lexical grabs the wrong memory) ==")
    shown = 0
    for q in queries:
        if shown >= 5:
            break
        if perq["embedder"][q["id"]] >= 0.99 and perq["lexical"][q["id"]] < 0.5:
            g = grades(q)
            primary = max(g, key=g.get)
            print(f"  {q['id']}: \"{q['text']}\"")
            print(f"     should recall: {primary}  ({mem_text[primary][:60]})")
            print(f"     embedder top1: {emb_ranked[q['id']][0]}  |  lexical top1: {lex_ranked[q['id']][0]}  ({mem_text[lex_ranked[q['id']][0]][:50]})")
            shown += 1


if __name__ == "__main__":
    main()
