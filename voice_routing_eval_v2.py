"""
HARDENED voice-routing eval. Ground truth comes from an LLM PANEL (majority vote of 3 different-family
models), independent of both routers and of my hand labels. Then heuristic vs embedding vs random are
scored against it. Also cross-checks: did my earlier hand labels agree with the panel?
"""
import json, random, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, str(Path.home() / "shopify_store"))
from dotenv import load_dotenv
load_dotenv(Path.home() / "shopify_store" / ".env")
from llm_router import _try_openrouter_one
from orlog.council.styles import ALL_STYLES

HERE = Path(__file__).parent
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
PANEL = ["meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat", "qwen/qwen-2.5-72b-instruct"]
CACHE = HERE / "voice_panel_labels.json"
NAMES = [s.name for s in ALL_STYLES]

def voice_menu():
    return "\n".join(f"- {s.name}: {(s.base_prompt or '')[:110]}" for s in ALL_STYLES)

PANEL_SYS = ("You assign cognitive VOICES to a decision situation. Given the situation and the list of voices "
             "with what each is for, return the 2 to 4 voices MOST relevant to this situation. Return JSON only: "
             '{"voices": ["name", ...]} using exact names from the list.')

def parse_voices(txt):
    if not txt: return []
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return []
    try:
        v = json.loads(m.group(0)).get("voices", [])
        return [x for x in v if x in NAMES]
    except Exception:
        return []

def panel_labels(situations):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    menu = voice_menu()
    for s in situations:
        t = s["text"]
        if t in cache: continue
        votes = {}
        for model in PANEL:
            user = f"VOICES:\n{menu}\n\nSITUATION:\n{t}\n\nWhich 2-4 voices are most relevant?"
            for v in parse_voices(_try_openrouter_one(PANEL_SYS, user, 80, model, 0.0)):
                votes[v] = votes.get(v, 0) + 1
        cache[t] = sorted([v for v, c in votes.items() if c >= 2])   # majority of 3
        CACHE.write_text(json.dumps(cache, indent=2))
    return cache

def recall_at_k(situations, gt, topk, k):
    rec = []
    for s in situations:
        rel = set(gt[s["text"]])
        if not rel: continue                      # skip situations the panel couldn't agree on
        rec.append(len(rel & set(topk[s["text"]][:k])) / len(rel))
    return float(np.mean(rec)), len(rec)

def heuristic(situations, k):
    return {s["text"]: [st.name for st in sorted(ALL_STYLES, key=lambda x: x.score_relevance(s["text"]), reverse=True)[:k]] for s in situations}

def embedding(situations, model, k):
    pv = np.asarray(model.encode([f"{s.name}: {s.base_prompt}" for s in ALL_STYLES], normalize_embeddings=True))
    out = {}
    for s in situations:
        qv = model.encode([s["text"]], normalize_embeddings=True)[0]
        out[s["text"]] = [NAMES[i] for i in np.argsort(-(pv @ qv))[:k]]
    return out

def main():
    d = json.load(open(HERE / "voice_routing_situations_v2.json"))
    situations, K = d["situations"], d["K"]
    print(f"panel-labeling {len(situations)} situations with {len(PANEL)} models (cached)...")
    gt = panel_labels(situations)
    model = SentenceTransformer(EMBEDDER)

    heur, emb = heuristic(situations, K), embedding(situations, model, K)
    rh, n = recall_at_k(situations, gt, heur, K)
    re_, _ = recall_at_k(situations, gt, emb, K)
    rr = float(np.mean([recall_at_k(situations, gt, {s["text"]: random.Random(seed).sample(NAMES, K) for s in situations}, K)[0] for seed in range(20)]))

    print(f"\n== VOICE ROUTING vs PANEL ground truth (recall@{K}, {n} scorable situations) ==")
    print(f"  random (control) : {rr:.2f}")
    print(f"  heuristic wheel  : {rh:.2f}")
    print(f"  embedding router : {re_:.2f}")
    print(f"  winner: {'embedding' if re_>rh else 'heuristic' if rh>re_ else 'tie'}  (both must beat random {rr:.2f})")

    # cross-check: did my hand labels agree with the panel, on the first 10?
    hand = [s for s in situations if "hand" in s]
    agree = tot = 0
    for s in hand:
        h, p = set(s["hand"]), set(gt[s["text"]])
        if p:
            agree += len(h & p); tot += len(h)
    print(f"\n  cross-check: my earlier hand labels overlap the panel on {agree}/{tot} = {100*agree/tot:.0f}% (sanity on my labeling)")

if __name__ == "__main__":
    main()
