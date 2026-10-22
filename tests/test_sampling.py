import torch
from latentspec.sampling import accept_reject, probs_from_logits, sample_rows


def test_lossless_first_token():
    torch.manual_seed(0)
    V, N, T = 6, 3, 60_000
    p = torch.softmax(torch.randn(N + 1, V), -1); q = torch.softmax(torch.randn(N, V), -1)
    counts = torch.zeros(V)
    for _ in range(T):
        d = torch.multinomial(q, 1).squeeze(1)
        n, y = accept_reject(p, q, d, torch.rand(N))
        counts[(d[0] if n > 0 else y)] += 1
    assert (counts / T - p[0]).abs().max() < 0.012   # first emitted token ~ p[0]


def test_greedy_accepts_only_matches():
    logits = torch.tensor([[0., 5, 0], [5., 0, 0], [0., 0, 5], [0., 5, 0]])
    p = probs_from_logits(logits, 0.0); q = p[:3].clone()
    n, y = accept_reject(p, q, torch.tensor([1, 0, 1]), torch.rand(3))
    assert int(n) == 2 and int(y) == 2            # third draft wrong -> corrected to argmax


def test_probs_from_logits_topk_topp():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pk = probs_from_logits(logits, 1.0, top_k=2)
    assert (pk > 0).sum() == 2
    assert torch.allclose(pk.sum(), torch.tensor(1.0), atol=1e-6)
    pp = probs_from_logits(logits, 1.0, top_p=0.5)
    assert (pp > 0).sum() <= 4 and torch.allclose(pp.sum(), torch.tensor(1.0), atol=1e-6)
    g = probs_from_logits(logits, 0.0, top_k=2, top_p=0.5)
    assert g.argmax(-1).item() == 3


def test_sample_rows_onehot():
    p = torch.tensor([0.0, 1.0, 0.0])
    s = sample_rows(p)
    assert int(s) == 1
