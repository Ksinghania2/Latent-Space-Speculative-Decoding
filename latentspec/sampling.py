"""Sync-free speculative sampling primitives (Leviathan et al. 2023 / Chen et al. 2023).

Greedy decoding (T=0) uses the same code path: p and q become one-hot, and the
accept/reject rule reduces to "accept iff argmax(p) == draft".
"""
import torch


def probs_from_logits(logits: torch.Tensor, temperature: float, top_k=None, top_p=None) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    # Optional truncation (applied identically to p and q to stay lossless).
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumprobs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return torch.softmax(logits / temperature, dim=-1)


def sample_rows(p: torch.Tensor) -> torch.Tensor:
    """Categorical sample per row via the exponential (Gumbel-max-equivalent) trick; no host sync."""
    e = torch.empty_like(p).exponential_().clamp_min_(1e-12)
    return (p / e).argmax(-1)


def accept_reject(p: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor, u: torch.Tensor):
    """Verify N drafted tokens against the target in one vectorised step.

    p: (N+1, V) target probs; row k scores draft k (row N is the bonus position).
    q: (N, V)   draft probs the tokens were sampled from.
    drafts: (N,) int64 draft tokens;  u: (N,) uniform[0,1).
    Returns (n_accepted: 0-d tensor, next_token: 0-d tensor). The emitted stream is
    distributed exactly as token-by-token sampling from the target (lossless).
    """
    N, V = q.shape
    idx = torch.arange(N, device=p.device)
    acc = (u * q[idx, drafts]) < p[idx, drafts]          # u < p/q without dividing
    n = torch.cumprod(acc.to(torch.int64), 0).sum()      # length of accepted prefix
    q_pad = torch.cat([q, q.new_zeros(1, V)], 0)         # zero row => bonus token ~ p[N]
    pn = p.index_select(0, n.view(1)).squeeze(0)
    qn = q_pad.index_select(0, n.view(1)).squeeze(0)
    res = (pn - qn).clamp_min(0)                         # residual distribution
    s = res.sum()
    res = torch.where(s > 1e-9, res / s.clamp_min(1e-30), pn)
    return n, sample_rows(res)
