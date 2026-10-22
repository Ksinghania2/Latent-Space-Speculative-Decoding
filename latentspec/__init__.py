"""Latent-Space Speculative Decoding: draft in hidden space, verify in one pass."""
from .draft_head import LatentDraftHead
from .engine import SpecEngine, autoregressive
from .sampling import accept_reject, probs_from_logits, sample_rows
