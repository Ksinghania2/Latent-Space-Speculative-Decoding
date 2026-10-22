"""Latent draft-and-verify loop.

One target forward pass per cycle does THREE jobs:
  1. verifies the N drafted tokens (rows of the final logits),
  2. yields the bonus / residual-corrected token,
  3. exposes the intermediate hidden state the draft head needs for the NEXT cycle,
so drafting costs one small MLP + one (N,d)@(d,V) matmul, never a second model pass.
"""
import time
import torch
from .sampling import accept_reject, probs_from_logits, sample_rows


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


class SpecEngine:
    def __init__(self, backbone, head, temperature=0.0, top_k=None, top_p=None, compile_modules=False):
        self.bb, self.T = backbone, temperature
        self.top_k, self.top_p = top_k, top_p
        self.head = head.to(backbone.device, backbone.dtype).eval()
        self.N = head.n_draft
        self._ar = accept_reject
        if compile_modules:  # fuses the small elementwise ops around draft/verify math
            self.head = torch.compile(self.head)
            self._ar = torch.compile(accept_reject, dynamic=False)

    @torch.no_grad()
    def _draft(self, h_row, y):
        e = self.bb.embed(y.view(1))
        z = self.head(h_row.view(1, -1), e)[0]                    # (N,d) latents, one pass
        q = probs_from_logits(self.bb.lm_head(z), self.T, self.top_k, self.top_p)         # decode latents -> (N,V)
        return sample_rows(q), q

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=16, eos_id=None, record=False):
        bb, dev, N = self.bb, self.bb.device, self.N
        _sync(dev); t0 = time.perf_counter()
        logits, hmid, _, cache = bb.forward(prompt_ids)
        c = prompt_ids.shape[1]
        y = sample_rows(probs_from_logits(logits[0, -1], self.T, self.top_k, self.top_p))
        h_row = hmid[0, -1]
        out, trace = [int(y)], []
        reached, accepted = [0] * N, [0] * N
        prefill_ms = (time.perf_counter() - t0) * 1e3
        while len(out) < max_new_tokens and (eos_id is None or out[-1] != eos_id):
            tc = time.perf_counter()
            drafts, q = self._draft(h_row, y)
            logits, hmid, _, cache = bb.forward(torch.cat([y.view(1), drafts]).unsqueeze(0), cache)
            p = probs_from_logits(logits[0], self.T, self.top_k, self.top_p)
            n, y_new = self._ar(p, q, drafts, torch.rand(N, device=dev))
            n_i = int(n)                                           # the single host sync per cycle
            bb.crop(cache, c + 1 + n_i)
            c += 1 + n_i
            for k in range(N):
                reached[k] += n_i >= k
                accepted[k] += n_i > k
            trace.append(dict(drafts=drafts.tolist(), accepted=n_i, final=int(y_new),
                              ms=(time.perf_counter() - tc) * 1e3))
            out += drafts[:n_i].tolist() + [int(y_new)]
            h_row, y = hmid[0, n_i], y_new
        _sync(dev)
        return dict(tokens=out[:max_new_tokens], ms=(time.perf_counter() - t0) * 1e3,
                    prefill_ms=prefill_ms, cycles=len(trace), reached=reached, accepted=accepted,
                    trace=trace if record else [])


@torch.no_grad()
def autoregressive(bb, prompt_ids, max_new_tokens=16, temperature=0.0, eos_id=None, top_k=None, top_p=None):
    _sync(bb.device); t0 = time.perf_counter()
    logits, _, _, cache = bb.forward(prompt_ids)
    out = [int(sample_rows(probs_from_logits(logits[0, -1], temperature, top_k, top_p)))]
    while len(out) < max_new_tokens and (eos_id is None or out[-1] != eos_id):
        logits, _, _, cache = bb.forward(torch.tensor([[out[-1]]], device=bb.device), cache)
        out.append(int(sample_rows(probs_from_logits(logits[0, -1], temperature, top_k, top_p))))
    _sync(bb.device)
    return dict(tokens=out, ms=(time.perf_counter() - t0) * 1e3)
