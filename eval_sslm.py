"""
The app-facing read on the converted Qwen3-Embedding .sslm (SuperEmbedder), next to float and the MiniLM
stand-in, on the SAME memories/queries/labels. Gives the "second read" to sit beside the converted model's own
fidelity number:
how much of FLOAT retrieval quality the quantized sslm keeps, measured application-side (recall the right
memory for a moment), not by-class.

  python3 eval_sslm.py
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/aioos_memory"))
import numpy as np
from eval import load, run_model, per_query_scores, bootstrap_ci, paired_diff_ci, K

KEY = "nDCG@%d" % K


def score(model, memories, queries):
    ranked, _, _ = run_model(model, memories, queries)
    return per_query_scores(queries, ranked)[KEY]


def main():
    memories, queries, pc = load()
    print(f"== app-facing eval ({len(queries)} queries, {len(memories)} memories, k={K}) ==\n")
    ndcg = {}

    from sentence_transformers import SentenceTransformer
    for label, mid in [("MiniLM (stand-in)", "sentence-transformers/all-MiniLM-L6-v2"),
                       ("Qwen3 FLOAT", "Qwen/Qwen3-Embedding-0.6B")]:
        print(f"  loading {label} ...")
        ndcg[label] = score(SentenceTransformer(mid), memories, queries)
        m, lo, hi = bootstrap_ci(ndcg[label])
        print(f"    {label:<18} {KEY} = {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print("  loading Qwen3 SSLM (SuperEmbedder, converted artifact) ...")
    from super_embedder import SuperEmbedder
    se = SuperEmbedder()
    ndcg["Qwen3 SSLM"] = score(se, memories, queries)
    m, lo, hi = bootstrap_ci(ndcg["Qwen3 SSLM"])
    print(f"    {'Qwen3 SSLM (dim '+str(se.dims)+')':<18} {KEY} = {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    se.close()

    fm = float(np.mean(list(ndcg["Qwen3 FLOAT"].values())))
    sm = float(np.mean(list(ndcg["Qwen3 SSLM"].values())))
    d, plo, phi, n = paired_diff_ci(ndcg["Qwen3 SSLM"], ndcg["Qwen3 FLOAT"])
    pct = 100 * sm / fm if fm else 0
    print(f"\n  SSLM vs FLOAT (the '% of float' read, cf. the converted model's own fidelity read ~95%):")
    print(f"    sslm keeps {pct:.0f}% of float quality ({sm:.3f} / {fm:.3f})")
    print(f"    paired diff sslm-float = {d:+.3f}  95% CI [{plo:+.3f}, {phi:+.3f}] (n={n})"
          f"  -> {'within noise (not resolvably worse = holds)' if plo <= 0 <= phi else ('sslm LOWER' if d<0 else 'sslm higher')}")


if __name__ == "__main__":
    main()
