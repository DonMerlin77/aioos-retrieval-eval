"""
VOICE-ROUTING EVAL — the rigorous version of "does Orlog pick the right voice?"
Routing a situation to the fitting voices IS a retrieval problem (same shape as memory retrieval).
This races three routers against hand-labeled ground truth (which voices SHOULD fire):
  heuristic  : Orlog's current wheel (score_relevance = keyword-trigger matching)
  embedding  : semantic retrieval (embed situation + each voice's profile, cosine)  <-- the proposed upgrade
  random     : control (must lose, or the metric is meaningless)
Metric: recall@K (of the voices that should fire, how many landed in the top K).
"""
import json, random
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, str(Path.home() / "shopify_store"))
from orlog.council.styles import ALL_STYLES

HERE = Path(__file__).parent
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

def load():
    d = json.load(open(HERE / "voice_routing_situations.json"))
    return d["situations"], d["K"]

def profile(style):
    # what we embed to represent a voice: its name + what it's for
    return f"{style.name}: {style.base_prompt}"

def recall_at_k(situations, router_topk, k):
    rec = []
    for s in situations:
        rel = set(s["relevant"])
        got = set(router_topk[s["text"]][:k])
        rec.append(len(rel & got) / len(rel))
    return float(np.mean(rec))

def heuristic_router(situations, k):
    out = {}
    for s in situations:
        scored = sorted(ALL_STYLES, key=lambda st: st.score_relevance(s["text"]), reverse=True)
        out[s["text"]] = [st.name for st in scored[:k]]
    return out

def embedding_router(situations, model, k):
    names = [st.name for st in ALL_STYLES]
    pvecs = np.asarray(model.encode([profile(st) for st in ALL_STYLES], normalize_embeddings=True))
    out = {}
    for s in situations:
        qv = model.encode([s["text"]], normalize_embeddings=True)[0]
        order = np.argsort(-(pvecs @ qv))
        out[s["text"]] = [names[i] for i in order[:k]]
    return out

def random_router(situations, k, seed):
    rng = random.Random(seed)
    names = [st.name for st in ALL_STYLES]
    return {s["text"]: rng.sample(names, k) for s in situations}

def main():
    situations, K = load()
    print(f"loading embedder: {EMBEDDER}")
    model = SentenceTransformer(EMBEDDER)

    heur = heuristic_router(situations, K)
    emb = embedding_router(situations, model, K)
    rand_scores = [recall_at_k(situations, random_router(situations, K, s), K) for s in range(20)]

    rh = recall_at_k(situations, heur, K)
    re = recall_at_k(situations, emb, K)
    rr = float(np.mean(rand_scores))
    print(f"\n== VOICE ROUTING — recall@{K} vs ground truth ({len(situations)} situations, {len(ALL_STYLES)} voices) ==")
    print(f"  random (control) : {rr:.2f}   <- must be low, or the task is trivial")
    print(f"  heuristic wheel  : {rh:.2f}")
    print(f"  embedding router : {re:.2f}")
    winner = "embedding" if re > rh else ("heuristic" if rh > re else "tie")
    print(f"\n  better router: {winner}   (both must beat random {rr:.2f} to mean anything)")

    print("\n  per-situation (want -> heuristic topK / embedding topK):")
    for s in situations:
        print(f"    {s['relevant']}")
        print(f"        heur: {heur[s['text']][:K]}")
        print(f"        emb : {emb[s['text']][:K]}")

if __name__ == "__main__":
    main()
