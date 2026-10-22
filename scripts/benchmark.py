"""AR vs latent speculative decoding: latency, acceptance, losslessness.
Writes results/bench.json and docs/trace.json (the web demo auto-loads the latter).
    python scripts/benchmark.py --head results/head.pt --max-new 16 --temp 0
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, statistics as st, torch
from latentspec import LatentDraftHead, SpecEngine, autoregressive
from latentspec.backbone import Backbone

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Explain why the sky is blue in one sentence:",
           "Once upon a time, in a small village", "The three laws of thermodynamics are", "import numpy as np\n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None); ap.add_argument("--head", default="results/head.pt")
    ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.head, map_location="cpu")
    bb = Backbone(a.model or ck["model"], mid_layer=ck["mid"])
    head = LatentDraftHead(**ck["cfg"]); head.load_state_dict(ck["state"])
    eng = SpecEngine(bb, head, a.temp, a.compile); N = head.n_draft
    enc = lambda s: bb.tok(s, return_tensors="pt").input_ids.to(bb.device)
    for _ in range(3):  # warm-up: CUDA context, allocator, compile
        eng.generate(enc(PROMPTS[0]), a.max_new); autoregressive(bb, enc(PROMPTS[0]), a.max_new, a.temp)
    rows, reached, acc, match, trace = [], [0] * N, [0] * N, [], None
    for pi, p in enumerate(PROMPTS):
        ids = enc(p)
        ar = st.median(autoregressive(bb, ids, a.max_new, a.temp)["ms"] for _ in range(a.runs))
        runs = [eng.generate(ids, a.max_new, record=True) for _ in range(a.runs)]
        sp = st.median(r["ms"] for r in runs); r = runs[0]
        if a.temp == 0:
            match.append(r["tokens"] == autoregressive(bb, ids, a.max_new, 0)["tokens"])
        for k in range(N):
            reached[k] += r["reached"][k]; acc[k] += r["accepted"][k]
        tau = (len(r["tokens"]) - 1) / max(1, r["cycles"])
        rows.append(dict(prompt=p, ar_ms=ar, spec_ms=sp, speedup=ar / sp, tokens_per_cycle=tau))
        print(f"{p[:30]!r:34} AR {ar:7.1f} ms | spec {sp:7.1f} ms | x{ar/sp:.2f} | tok/cycle {tau:.2f}")
        if pi == 0:
            d = lambda i: bb.tok.decode([i])
            trace = dict(model=ck["model"], prompt=p, n_draft=N, first=d(r["tokens"][0]),
                         ar_ms_per_token=ar / a.max_new,
                         cycles=[dict(drafts=[d(t) for t in c["drafts"]], accepted=c["accepted"],
                                      final=d(c["final"]), ms=c["ms"]) for c in r["trace"]])
    summary = dict(model=ck["model"], n_draft=N, temp=a.temp, max_new=a.max_new,
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                   mean_ar_ms=st.mean(r["ar_ms"] for r in rows), mean_spec_ms=st.mean(r["spec_ms"] for r in rows),
                   mean_speedup=st.mean(r["speedup"] for r in rows),
                   alpha_by_offset=[acc[k] / max(1, reached[k]) for k in range(N)],
                   lossless_greedy=(all(match) if match else None), rows=rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    json.dump(summary, open("results/bench.json", "w"), indent=2)
    json.dump(trace, open("docs/trace.json", "w"), indent=2)


if __name__ == "__main__":
    main()
