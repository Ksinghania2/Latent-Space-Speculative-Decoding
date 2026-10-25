import pytest
import torch
from torch import nn

from latentspec.engine import SpecEngine, autoregressive


class FixedLMHead(nn.Module):
    def forward(self, hidden):
        logits = torch.full((*hidden.shape[:-1], 3), -10.0, device=hidden.device)
        logits[..., 2] = 10.0
        return logits


class FixedDraftHead(nn.Module):
    n_draft = 3

    def forward(self, h_mid, e_next):
        return h_mid.new_zeros((h_mid.shape[0], self.n_draft, h_mid.shape[-1]))


class FixedBackbone:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.lm_head = FixedLMHead()
        self.calls = 0

    def embed(self, token):
        return torch.zeros((token.numel(), 1), device=self.device)

    def forward(self, ids, cache=None):
        vocab_token = 1 if self.calls == 0 else 2
        self.calls += 1
        logits = torch.full((*ids.shape, 3), -10.0, device=self.device)
        logits[..., vocab_token] = 10.0
        hidden = torch.zeros((*ids.shape, 1), device=self.device)
        return logits, hidden, hidden, None

    def crop(self, cache, length):
        pass


def test_speculative_generation_stops_on_eos_inside_accepted_drafts():
    backbone = FixedBackbone()
    engine = SpecEngine(backbone, FixedDraftHead(), temperature=0.0)
    result = engine.generate(torch.tensor([[0]]), max_new_tokens=8, eos_id=2, record=True)
    assert result["tokens"] == [1, 2]
    assert result["trace"][0]["emitted"] == 1


def test_speculative_generation_respects_token_limit_inside_cycle():
    backbone = FixedBackbone()
    engine = SpecEngine(backbone, FixedDraftHead(), temperature=0.0)
    result = engine.generate(torch.tensor([[0]]), max_new_tokens=2, record=True)
    assert result["tokens"] == [1, 2]
    assert result["trace"][0]["emitted"] == 1


def test_generation_with_zero_tokens_does_not_run_backbone():
    backbone = FixedBackbone()
    engine = SpecEngine(backbone, FixedDraftHead(), temperature=0.0)
    result = engine.generate(torch.tensor([[0]]), max_new_tokens=0)
    assert result["tokens"] == []
    assert backbone.calls == 0
    assert autoregressive(backbone, torch.tensor([[0]]), max_new_tokens=0)["tokens"] == []


def test_generation_rejects_negative_token_limit():
    backbone = FixedBackbone()
    engine = SpecEngine(backbone, FixedDraftHead(), temperature=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        engine.generate(torch.tensor([[0]]), max_new_tokens=-1)
    with pytest.raises(ValueError, match="non-negative"):
        autoregressive(backbone, torch.tensor([[0]]), max_new_tokens=-1)
