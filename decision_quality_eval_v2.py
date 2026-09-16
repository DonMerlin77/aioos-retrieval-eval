"""
HARDENED decision-quality test: single voice vs 3 routed voices, on calibration-hard decisions.
  - each decision run R times per arm (average out temperature noise)
  - robust positive control: 5 trials/arm, BOTH must pass >=4/5 or the whole result is VOID
  - paired McNemar on per-decision majority (did routed fix what single missed, vs the reverse)
Parallelized. Objective metric (fact-determined correct action), out-of-family judge (deepseek).
"""
import json, random, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, str(Path.home() / "shopify_store"))
from dotenv import load_dotenv
load_dotenv(Path.home() / "shopify_store" / ".env")
from llm_router import _try_openrouter_one
from orlog.council.styles import ALL_STYLES

HERE = Path(__file__).parent
ARM = "meta-llama/llama-3.3-70b-instruct"
JUDGE = "deepseek/deepseek-chat"
R = 3
_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_pv = np.asarray(_emb.encode([f"{s.name}: {s.base_prompt}" for s in ALL_STYLES], normalize_embeddings=True))
TAG = "\nEnd with exactly one line: 'DECISION: <your final call in a short phrase>'."

def blk(m): return "\n".join(f"- {x}" for x in m)

def single(d):
    s = ("You are a tavern keeper making a real decision. Weigh your memories by recency, trustworthiness, and "
         "corroboration. Commit to a clear call in 2-3 sentences, naming any real doubt." + TAG)
    return _try_openrouter_one(s, f"SITUATION:\n{d['situation']}\n\nMEMORIES:\n{blk(d['memories'])}\n\nWhat do you do?", 160, ARM, 0.4) or "[none]"

def routed3(d):
    voices = [ALL_STYLES[i] for i in np.argsort(-(_pv @ _emb.encode([d["situation"]], normalize_embeddings=True)[0]))[:3]]
    takes = []
    for v in voices:
        vs = f"{v.base_prompt}\nYou are ONE advisor. Give your take in 2 sentences from your angle only."
        takes.append(f"[{v.name}] " + (_try_openrouter_one(vs, f"SITUATION:\n{d['situation']}\n\nMEMORIES:\n{blk(d['memories'])}", 90, ARM, 0.4) or ""))
    s = ("You are the decision-maker. Weigh the memories by recency, trust, and corroboration given your advisors' "
         "takes, and commit to a clear call in 2-3 sentences, naming any real doubt." + TAG)
    u = f"SITUATION:\n{d['situation']}\n\nMEMORIES:\n{blk(d['memories'])}\n\nADVISORS:\n" + "\n".join(takes) + "\n\nFinal call?"
    return _try_openrouter_one(s, u, 160, ARM, 0.3) or "[none]"

def judge(d, answer):
    ca = random.random() < 0.5
    A, B = (d["correct"], d["wrong"]) if ca else (d["wrong"], d["correct"])
    r = _try_openrouter_one('Which action did the ANSWER take? JSON {"choice":"A"|"B"|"neither"}.',
                            f"SITUATION:\n{d['situation']}\n\nANSWER:\n{answer[:1200]}\n\nA: {A}\nB: {B}", 20, JUDGE, 0.0)
    m = re.search(r'"choice"\s*:\s*"(A|B|neither)"', r or "")
    if not m or m.group(1) == "neither": return "none"
    return "correct" if ((m.group(1) == "A") == ca) else "wrong"

ARMS = {"single": single, "routed3": routed3}

def run_units(decisions, reps):
    tasks = [(d, arm, t) for d in decisions for arm in ARMS for t in range(reps)]
    out = {arm: {d["id"]: [] for d in decisions} for arm in ARMS}
    def work(task):
        d, arm, _ = task
        return d["id"], arm, judge(d, ARMS[arm](d))
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(work, t) for t in tasks]):
            did, arm, verdict = f.result()
            out[arm][did].append(verdict)
    return out

def main():
    data = json.load(open(HERE / "decisions.json"))
    pc = data["positive_control"]

    # robust positive control: 5 trials/arm, both must pass >=4/5
    print("robust positive control (5 trials/arm, both need >=4/5):")
    valid = True
    for arm, fn in ARMS.items():
        vs = [judge(pc, fn(pc)) for _ in range(5)]
        n_ok = vs.count("correct")
        print(f"  {arm}: {n_ok}/5 correct")
        valid = valid and n_ok >= 4
    if not valid:
        print("\n  CONTROL FAILED -> result VOID, the metric is not reliable. Not reporting numbers.")
        return

    out = run_units(data["decisions"], R)
    print(f"\n== DECISION QUALITY (n={len(data['decisions'])} decisions x {R} reps, control passed) ==")
    accs = {}
    for arm in ARMS:
        allv = [v for did in out[arm] for v in out[arm][did]]
        accs[arm] = allv.count("correct") / len(allv)
        print(f"  {arm:<9}: {accs[arm]*100:.0f}%  (correct action over all trials)")

    # paired McNemar on per-decision majority
    from math import comb
    b = c = 0
    for d in data["decisions"]:
        sm = out["single"][d["id"]].count("correct") >= (R//2 + 1)
        rm = out["routed3"][d["id"]].count("correct") >= (R//2 + 1)
        if rm and not sm: b += 1
        elif sm and not rm: c += 1
    nn = b + c
    p = 1.0 if nn == 0 else min(1.0, 2 * sum(comb(nn, k) for k in range(min(b, c)+1)) / 2**nn)
    print(f"\n  paired: routed fixed {b} that single missed; single fixed {c} that routed missed; McNemar p={p:.3f}")
    print(f"  verdict: routed {'>' if accs['routed3']>accs['single'] else '<' if accs['routed3']<accs['single'] else '='} single")

if __name__ == "__main__":
    main()
