"""
Missed-label audit -- closes the one validity hole the pair-audit (audit_labels.py) leaves open.

audit_labels.py only re-grades pairs we ALREADY labeled. It cannot catch a memory that is
genuinely relevant to a query but was never labeled at all -- a false negative in the labels. If
the embedder retrieves such a memory, the eval wrongly counts it against the embedder (it scores
as a 0). That would make the benchmark unfairly hard and, worse, invalid.

This audits exactly the memories at risk of that: for each query, the memories the STAND-IN
embedder ranked into the top-k that are NOT labeled and are NOT the deliberate hard_negative. An
independent judge grades each; anything it calls relevant (grade >= 1) is a candidate MISSED label
for a human to adjudicate. Targeted, so it is cheap -- only the retrieved-but-uncredited memories,
which are the only ones that can actually distort the score.

Run:  python audit_missed.py        (writes missed_flags.json)
"""
import json, re, sys
from pathlib import Path
from eval import load, run_model, MODEL, K
from audit_labels import judge
from sentence_transformers import SentenceTransformer

TOPK = 5  # audit a little deeper than the deciding k=3, to catch near-misses too

def main():
    only = {a for a in sys.argv[1:] if re.fullmatch(r"q\d+", a)}  # audit only these query ids if given
    memories, queries, _ = load()
    if only:
        queries = [q for q in queries if q["id"] in only]
    mem = {m["id"]: m["text"] for m in memories}
    print(f"loading embedder: {MODEL}")
    model = SentenceTransformer(MODEL)
    ranked, _, _ = run_model(model, memories, queries)

    flags, checked = [], 0
    for q in queries:
        labeled = set(q["labels"])
        hn = q.get("hard_negative")
        candidates = [m for m in ranked[q["id"]][:TOPK] if m not in labeled and m != hn]
        for mid in candidates:
            jg, reason = judge(q["text"], mem[mid])
            checked += 1
            if jg is not None and jg >= 1:
                flags.append({"query": q["id"], "memory": mid, "judge_grade": jg,
                              "judge_reason": reason, "situation": q["text"], "memory_text": mem[mid]})

    Path("missed_flags.json").write_text(json.dumps(flags, indent=2))
    print(f"judged {checked} retrieved-but-unlabeled memories")
    print(f"{len(flags)} candidate MISSED label(s) -> missed_flags.json\n")
    for f in flags:
        print(f"  [{f['query']}/{f['memory']}] judge={f['judge_grade']}: {f['situation'][:55]}")
        print(f"      mem: {f['memory_text'][:70]}  ({f['judge_reason'][:60]})")
    if not flags:
        print("  none -- the embedder retrieved nothing relevant that the labels missed. Labels are complete w.r.t. what it surfaces.")

if __name__ == "__main__":
    main()
