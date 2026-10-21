"""
Head-to-head: the MiniLM stand-in vs the real Qwen3-0.6B embedder, on the SAME memories/queries/labels.
Uses the eval's own paired-difference CI so the read is honest: if the CI straddles 0 the two are NOT
resolvably different on this eval (report a tie), not a fake win.

  python3 compare_embedders.py
"""
from sentence_transformers import SentenceTransformer
from eval import load, run_model, per_query_scores, bootstrap_ci, paired_diff_ci, executed_chance, K

MODELS = [
    ("MiniLM (stand-in)", "sentence-transformers/all-MiniLM-L6-v2"),
    ("Qwen3-0.6B (real)", "Qwen/Qwen3-Embedding-0.6B"),
]
KEY = "nDCG@%d" % K


def main():
    memories, queries, pc = load()
    mem_ids = [m["id"] for m in memories]
    ch_mean, ch_std = executed_chance(queries, mem_ids, K)
    print(f"== embedder head-to-head ({len(queries)} queries, {len(memories)} memories, k={K}) ==")
    print(f"   random-ranking chance floor: nDCG@{K} = {ch_mean:.3f} +/- {ch_std:.3f}\n")

    ndcg = {}
    for label, mid in MODELS:
        print(f"  loading {label}: {mid} ...")
        model = SentenceTransformer(mid)
        ranked, _, _ = run_model(model, memories, queries)
        scores = per_query_scores(queries, ranked)
        ndcg[label] = scores[KEY]
        mean, lo, hi = bootstrap_ci(scores[KEY])
        # positive control: a near-copy of a memory must retrieve that exact memory at rank 1
        pcv = model.encode([pc["text"]], normalize_embeddings=True)[0]
        import numpy as np
        mvecs = model.encode([m["text"] for m in memories], normalize_embeddings=True)
        pc_top = mem_ids[int(np.argmax(np.asarray(mvecs) @ np.asarray(pcv)))]
        pc_ok = pc_top == pc["must_retrieve"]
        print(f"    {label}: {KEY} = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   positive control: {'PASS' if pc_ok else 'FAIL'}\n")

    a, b = "Qwen3-0.6B (real)", "MiniLM (stand-in)"
    d, lo, hi, n = paired_diff_ci(ndcg[a], ndcg[b])
    resolved = lo > 0 or hi < 0
    print(f"  PAIRED  {a} - {b}  =  {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   (n={n} shared queries)")
    if not resolved:
        print(f"  -> TIED: the CI straddles 0, so the two are NOT resolvably different on this {n}-query eval.")
        print(f"     (honest read -- a bigger/harder query set is what would resolve them; same lesson as K1.)")
    else:
        winner = a if d > 0 else b
        print(f"  -> RESOLVABLE: {winner} is better here (CI clears 0).")


if __name__ == "__main__":
    main()
