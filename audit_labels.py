"""
Independent-judge label audit. Separates label GENERATION from label CHECKING: a different model
(not the embedder, not the drafter) scores each candidate (query, memory) pair for relevance under
the SAME rubric, and flags where it disagrees with the draft grade in memories.json. Tom then
adjudicates only the flagged pairs -- his ruling is the ground truth of record.

This is the anti-Goodhart control for the LABELS themselves (the same shape as the reasoning
project's separated eliminator): the thing that decides relevance must not be the thing that
proposed it.

Default pass audits every drafted pair + each query's hard_negative (cheap, ~90 calls).
--full also sweeps every memory against every query to catch MISSED relevant memories the draft
forgot to label (expensive, ~n_queries * n_memories calls).

  OPENROUTER_API_KEY loaded from ~/shopify_store/.env (Tom's convention).
  AUDIT_MODEL overrides the judge model (default: an instruct model distinct from the embedder).

Run:  python audit_labels.py          # validate drafted grades + hard negatives
      python audit_labels.py --full   # also hunt for missed-relevant memories
"""
import json, os, re, sys, time
from pathlib import Path
import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(Path.home() / "shopify_store" / ".env")
API_KEY = os.getenv("OPENROUTER_API_KEY")
AUDIT_MODEL = os.getenv("AUDIT_MODEL", "meta-llama/llama-3.3-70b-instruct")
URL = "https://openrouter.ai/api/v1/chat/completions"

RUBRIC = (
    "You are grading how relevant ONE memory is to ONE situation an NPC tavern keeper is in. "
    "Grade strictly:\n"
    "2 = PRIMARY: a response that fails to recall this memory would be wrong or clueless.\n"
    "1 = SUPPORTING: genuinely relevant and worth surfacing, but a response is competent without it.\n"
    "0 = NOT relevant: off-topic, or only superficially similar (same subject, wrong substance).\n"
    "Judge substance, not word overlap. Reply with exactly: GRADE=<0|1|2> | <one short reason>."
)


def judge(situation, memory_text, retries=4):
    body = {
        "model": AUDIT_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"SITUATION: {situation}\nMEMORY: {memory_text}\n\nGrade this memory."},
        ],
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=90)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"GRADE\s*=\s*([012])", txt)
            grade = int(m.group(1)) if m else None
            reason = txt.split("|", 1)[1].strip() if "|" in txt else txt
            return grade, reason
        except Exception as e:  # a flaky network/API call must not abort the whole batch
            last = e
            time.sleep(2 * (attempt + 1))
    return None, f"REQUEST FAILED after {retries} tries: {last}"


def main():
    full = "--full" in sys.argv
    only = {a for a in sys.argv[1:] if re.fullmatch(r"q\d+", a)}  # audit only these query ids if given
    if not API_KEY:
        sys.exit("OPENROUTER_API_KEY not found in ~/shopify_store/.env")
    d = json.load(open(HERE / "memories.json"))
    mem = {m["id"]: m["text"] for m in d["memories"]}
    queries = [q for q in d["queries"] if not only or q["id"] in only]
    flags, checked = [], 0

    for q in queries:
        draft = {mid: lab["grade"] for mid, lab in q["labels"].items()}
        candidates = dict(draft)
        if q.get("hard_negative"):
            candidates.setdefault(q["hard_negative"], 0)  # drafted as a deliberate 0
        if full:
            for mid in mem:
                candidates.setdefault(mid, 0)

        for mid, draft_grade in candidates.items():
            jg, reason = judge(q["text"], mem[mid])
            checked += 1
            if jg is None:
                flags.append({"query": q["id"], "memory": mid, "issue": "unparsed", "judge_raw": reason})
            elif jg != draft_grade:
                kind = ("MISSED-RELEVANT" if draft_grade == 0 and jg > 0
                        else "OVER-LABELED" if jg == 0 and draft_grade > 0
                        else "GRADE-MISMATCH")
                flags.append({"query": q["id"], "memory": mid, "issue": kind,
                              "draft": draft_grade, "judge": jg, "judge_reason": reason,
                              "situation": q["text"], "memory_text": mem[mid]})
            time.sleep(0.2)

    (HERE / "audit_flags.json").write_text(json.dumps(flags, indent=2))
    print(f"judged {checked} (query, memory) pairs with {AUDIT_MODEL}")
    print(f"{len(flags)} disagreement(s) for Tom to adjudicate -> audit_flags.json\n")
    for f in flags:
        if f["issue"] == "unparsed":
            print(f"  [{f['query']}/{f['memory']}] UNPARSED: {f['judge_raw'][:80]}")
        else:
            print(f"  [{f['query']}/{f['memory']}] {f['issue']}: draft={f['draft']} judge={f['judge']}  ({f['judge_reason'][:80]})")
    if not flags:
        print("  no disagreements -- draft labels survive the independent judge (still spot-check by hand).")


if __name__ == "__main__":
    main()
