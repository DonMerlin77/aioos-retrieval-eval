"""
COVERAGE test — the FAIR test of Orlog's real value. On open analytical questions, of the KEY
considerations a thorough answer should raise, how many does each approach surface?
  single      : one good prompt
  orlog_synth : the REAL Orlog synthesis (RoundTable.deliberate), the verdict+reasoning
  orlog_voices: the union of the raw voice perspectives (before synthesis)
A judge (deepseek) decides, per consideration, whether the answer raises it. This measures the
'surface every angle' value objectively, instead of 'pick the correct action'.
"""
import json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
ROOT = Path.home() / "shopify_store"
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from llm_router import _try_openrouter_one
from orlog.council import council as C

ARM_MODEL = "meta-llama/llama-3.3-70b-instruct"
JUDGE = "deepseek/deepseek-chat"
HERE = Path(__file__).parent

# make the real Orlog run on a reliable model (its free stack stalls)
def _call(self, system, user, max_tokens=400, agent_tag="orlog_council"):
    return _try_openrouter_one(system, user, max_tokens, ARM_MODEL, 0.4) or "[empty]"
C.RoundTable._call = _call

def single(q):
    s = "You are a sharp advisor. Think through this and give your best analysis in 5-8 sentences."
    return _try_openrouter_one(s, q, 400, ARM_MODEL, 0.4) or ""

def orlog(q):
    r = C.RoundTable().deliberate(q, max_voices=6)
    synth = r.get("synthesis", "")
    voices = "\n".join(f"[{k}] {v}" for k, v in r.get("perspectives", {}).items())
    return synth, voices

def raises(answer, consideration):
    r = _try_openrouter_one(
        'Does the ANSWER meaningfully raise or address this CONSIDERATION? JSON {"raised": true|false}.',
        f"CONSIDERATION: {consideration}\n\nANSWER:\n{answer[:2500]}", 15, JUDGE, 0.0)
    m = re.search(r'"raised"\s*:\s*(true|false)', r or "")
    return bool(m and m.group(1) == "true")

def main():
    qs = json.load(open(HERE / "open_questions.json"))["questions"]
    arms = {}
    print("generating answers (single + real Orlog)...")
    for q in qs:
        s = single(q["text"]); syn, voi = orlog(q["text"])
        arms[q["id"]] = {"single": s, "orlog_synth": syn, "orlog_voices": voi}
        print(f"  {q['id']} done")

    # judge coverage in parallel
    jobs = [(q["id"], arm, c) for q in qs for arm in ("single", "orlog_synth", "orlog_voices") for c in q["considerations"]]
    cov = {a: [] for a in ("single", "orlog_synth", "orlog_voices")}
    def work(j):
        qid, arm, c = j
        return arm, raises(arms[qid][arm], c)
    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            arm, hit = f.result(); cov[arm].append(hit)

    total = sum(len(q["considerations"]) for q in qs)
    print(f"\n== CONSIDERATION COVERAGE ({total} key considerations across {len(qs)} open questions) ==")
    for arm in ("single", "orlog_synth", "orlog_voices"):
        c = sum(cov[arm])
        print(f"  {arm:<13}: {c}/{total} = {100*c/total:.0f}%")
    print("\n  (does the real Orlog surface MORE of the key angles than a single pass? that's its actual claim.)")

if __name__ == "__main__":
    main()
