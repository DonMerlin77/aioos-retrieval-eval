"""
DECISION-QUALITY test: single voice vs 3 ROUTED voices, on calibration-hard decisions with a
FACT-DETERMINED correct action (objective, un-Goodhartable). Does fixing the routing make a small
council at least TIE a single prompt? (The blunt council lost ~29/69; this is the routed version.)

Arms (same model):
  single  : one prompt, situation + memories -> decision
  routed3 : embedding-route to the 3 fittest voices -> each gives a take -> synthesize -> decision
Scoring: an out-of-family judge (deepseek) says whether the answer took the CORRECT or the WRONG action.
Controls: a positive control both arms must pass; if not, the test is broken.
"""
import json, random
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
ARM_MODEL = "meta-llama/llama-3.3-70b-instruct"
JUDGE = "deepseek/deepseek-chat"
NAMES = [s.name for s in ALL_STYLES]
_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_pvecs = np.asarray(_emb.encode([f"{s.name}: {s.base_prompt}" for s in ALL_STYLES], normalize_embeddings=True))

def block(mems):
    return "\n".join(f"- {m}" for m in mems)

DECISION_TAG = "\nEnd with exactly one line: 'DECISION: <your final call in a short phrase>'."

def single(d):
    sys_ = ("You are a tavern keeper making a real decision. Weigh your memories by how recent, trustworthy, and "
            "corroborated they are. Commit to a clear call in 2-3 sentences, naming any real doubt." + DECISION_TAG)
    u = f"SITUATION:\n{d['situation']}\n\nWHAT YOU REMEMBER:\n{block(d['memories'])}\n\nWhat do you do?"
    return _try_openrouter_one(sys_, u, 160, ARM_MODEL, 0.3) or "[none]"

def route3(situation):
    qv = _emb.encode([situation], normalize_embeddings=True)[0]
    return [ALL_STYLES[i] for i in np.argsort(-(_pvecs @ qv))[:3]]

def routed3(d):
    voices = route3(d["situation"])
    takes = []
    for v in voices:
        vs = f"{v.base_prompt}\nYou are ONE advisor. Give your take in 2 sentences, from your angle only."
        u = f"SITUATION:\n{d['situation']}\n\nMEMORIES:\n{block(d['memories'])}"
        takes.append(f"[{v.name}] " + (_try_openrouter_one(vs, u, 90, ARM_MODEL, 0.4) or ""))
    syn = ("You are the decision-maker. Given your advisors' takes, weigh the memories by recency, trust, and "
           "corroboration, and commit to a clear call in 2-3 sentences, naming any real doubt." + DECISION_TAG)
    u = f"SITUATION:\n{d['situation']}\n\nMEMORIES:\n{block(d['memories'])}\n\nADVISORS:\n" + "\n".join(takes) + "\n\nFinal call?"
    return _try_openrouter_one(syn, u, 160, ARM_MODEL, 0.3) or "[none]"

def judge_action(d, answer):
    correct_is_A = random.random() < 0.5
    A, B = (d["correct"], d["wrong"]) if correct_is_A else (d["wrong"], d["correct"])
    sys_ = ('Which action did the ANSWER actually take? Return JSON {"choice":"A"|"B"|"neither"}.')
    u = f"SITUATION:\n{d['situation']}\n\nANSWER:\n{answer[:1200]}\n\nACTION A: {A}\nACTION B: {B}"
    import re
    r = _try_openrouter_one(sys_, u, 20, JUDGE, 0.0)
    m = re.search(r'"choice"\s*:\s*"(A|B|neither)"', r or "")
    if not m:
        return "none"
    if m.group(1) == "neither":
        return "none"
    picked_A = m.group(1) == "A"
    return "correct" if (picked_A == correct_is_A) else "wrong"

def acc(results):
    n = len(results); c = sum(1 for x in results if x == "correct")
    return c, n

def main():
    data = json.load(open(HERE / "decisions.json"))
    pc = data["positive_control"]
    print("positive control (both arms MUST get right):")
    for name, fn in [("single", single), ("routed3", routed3)]:
        v = judge_action(pc, fn(pc))
        print(f"  {name}: {v}")
    print()

    single_res, routed_res = [], []
    print("per-decision (single / routed3):")
    for d in data["decisions"]:
        s = judge_action(d, single(d)); r = judge_action(d, routed3(d))
        single_res.append(s); routed_res.append(r)
        print(f"  {d['id']}: single={s:<8} routed3={r}")
    sc, n = acc(single_res); rc, _ = acc(routed_res)
    print(f"\n== DECISION QUALITY (correct action, n={n}) ==")
    print(f"  single voice     : {sc}/{n} = {100*sc/n:.0f}%")
    print(f"  3 routed voices  : {rc}/{n} = {100*rc/n:.0f}%")
    print(f"  verdict: routed {'>' if rc>sc else '<' if rc<sc else '='} single")

if __name__ == "__main__":
    main()
