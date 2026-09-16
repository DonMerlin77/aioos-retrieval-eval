"""
AIOOS living world: a tick loop where the NPC FORMS memories from events and its behavior EVOLVES.
Memory is mutable and grows during play (SuperFAISS's runtime-bank feature; numpy here for the prototype).

The demo/test: the SAME question is asked at two points in time. Before the NPC learns something, and
after. If memory-writing works, the answer CHANGES appropriately. If it doesn't, the two answers match
and the whole memory stack is decorative.

  python3 world.py            # runs the scripted day
"""
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, str(Path.home() / "shopify_store"))
from dotenv import load_dotenv
load_dotenv(Path.home() / "shopify_store" / ".env")
from llm_router import _try_openrouter_one

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
DECIDE_MODEL = os.getenv("MODEL", "meta-llama/llama-3.3-70b-instruct")
NPC_SYS = ("You are Bran, keeper of a small-town tavern. Stay in character. Answer in 2-3 sentences, grounded "
           "ONLY in the memories you are given. Do not invent facts you were not told.")

class NPC:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDER)
        self.memories = []                       # grows during play
        self._vecs = None

    def observe(self, event):                    # MEMORY WRITING: turn an event into a stored memory
        mid = f"m{len(self.memories)+1}"
        self.memories.append({"id": mid, "text": event})
        self._vecs = None                        # invalidate cache; bank grew
        return mid

    def _bank(self):
        if self._vecs is None and self.memories:
            self._vecs = np.asarray(self.model.encode([m["text"] for m in self.memories], normalize_embeddings=True))
        return self._vecs

    def retrieve(self, situation, k=3):
        if not self.memories:
            return []
        qv = self.model.encode([situation], normalize_embeddings=True)[0]
        order = np.argsort(-(self._bank() @ qv))
        return [self.memories[i] for i in order[:k]]

    def decide(self, situation, k=3):
        recalled = self.retrieve(situation, k)
        block = "\n".join(f"- {m['text']}" for m in recalled) or "(you recall nothing relevant)"
        user = f"SITUATION:\n{situation}\n\nWHAT YOU REMEMBER:\n{block}\n\nWhat do you say?"
        resp = _try_openrouter_one(NPC_SYS, user, 180, DECIDE_MODEL, 0.4) or "[no response]"
        return resp.strip(), [m["id"] for m in recalled]

TIMELINE = [
    ("event", "You overhear two guards saying the north bridge collapsed in last night's storm."),
    ("event", "A merchant paid with a coin bearing the royal seal, then left in a hurry."),
    ("situation", "A traveler asks you the safest route to take heading north."),      # should warn: bridge out
    ("event", "Word arrives that the north bridge has been fully repaired and the road is open again."),
    ("event", "The mayor announces the harvest festival is moved to next week."),
    ("situation", "Another traveler asks you the safest route to take heading north."),  # should now say: road is fine
]

def main():
    bran = NPC()
    print(f"decision model: {DECIDE_MODEL}\n" + "=" * 78)
    for kind, text in TIMELINE:
        if kind == "event":
            mid = bran.observe(text)
            print(f"\n[EVENT -> stored as {mid}, memory now {len(bran.memories)}] {text}")
        else:
            resp, recalled = bran.decide(text)
            print(f"\n[SITUATION] {text}")
            print(f"   recalled: {recalled}")
            print(f"   Bran: {resp}")
    print("\n" + "=" * 78)
    print("THE TEST: the two 'route north' answers should DIFFER, because Bran learned the bridge was repaired\n"
          "between them. Same question, different answer = memory-writing changed behavior over time.")

if __name__ == "__main__":
    main()
