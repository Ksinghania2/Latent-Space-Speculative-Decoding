"""Terminal demo: python scripts/demo.py --head results/head.pt   (green = accepted drafts, blue = corrected/bonus token)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, torch
from latentspec import LatentDraftHead, SpecEngine
from latentspec.backbone import Backbone

ap = argparse.ArgumentParser(); ap.add_argument("--head", default="results/head.pt")
ap.add_argument("--max-new", type=int, default=40); ap.add_argument("--temp", type=float, default=0.0)
a = ap.parse_args()
ck = torch.load(a.head, map_location="cpu"); bb = Backbone(ck["model"], mid_layer=ck["mid"])
head = LatentDraftHead(**ck["cfg"]); head.load_state_dict(ck["state"]); eng = SpecEngine(bb, head, a.temp)
G, B, R = "\033[92m", "\033[94m", "\033[0m"
while True:
    try:
        prompt = input("\nprompt> ")
    except EOFError:
        break
    r = eng.generate(bb.tok(prompt, return_tensors="pt").input_ids.to(bb.device), a.max_new, record=True)
    print(bb.tok.decode(r["tokens"][:1]), end="")
    for c in r["trace"]:
        accepted = min(c["accepted"], c["emitted"])
        print(G + bb.tok.decode(c["drafts"][:accepted]), end="", flush=True)
        if c["emitted"] > c["accepted"]:
            print(B + bb.tok.decode([c["final"]]), end="", flush=True)
        print(R, end="")
    print(f"\n[{r['ms']:.1f} ms | {len(r['tokens'])} tokens | {r['cycles']} verify passes | "
          f"{(len(r['tokens'])-1)/max(1,r['cycles']):.2f} tokens/pass]")
