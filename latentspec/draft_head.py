import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


class LatentDraftHead(nn.Module):
    """Predicts the final-layer latents of the next N positions in ONE forward pass.

    Input : h_mid  (B, d) intermediate-layer hidden state at the last accepted position
            e_next (B, d) embedding of the already-committed next token y
    Output: z      (B, N, d) with z[:, k] ~ final (post-norm) hidden state at position(y)+k.
    Draft token k = argmax/sample of lm_head(z[:, k]) using the FROZEN LM head, so all N
    logits come from a single (N,d)@(d,V) matmul = one read of the head weights.
    """

    def __init__(self, d_model: int, n_draft: int = 4, expansion: int = 2):
        super().__init__()
        h = d_model * expansion
        self.d_model, self.n_draft, self.expansion = d_model, n_draft, expansion
        self.n_h, self.n_e = RMSNorm(d_model), RMSNorm(d_model)
        self.fuse = nn.Linear(2 * d_model, h)
        self.mix = nn.Sequential(nn.SiLU(), nn.Linear(h, h))
        self.step = nn.Parameter(torch.randn(n_draft, h) * 0.02)   # per-offset conditioning
        self.out = nn.Linear(h, d_model)
        self.skip = nn.Linear(d_model, d_model, bias=False)        # direct mid->latent path
        # second residual path from the fused feature, helps gradient flow
        self.res = nn.Linear(h, d_model, bias=False)
        nn.init.xavier_uniform_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.xavier_uniform_(self.res.weight)

    def forward(self, h_mid, e_next):
        hn = self.n_h(h_mid)
        en = self.n_e(e_next)
        u = self.mix(self.fuse(torch.cat([hn, en], -1)))
        base = u.unsqueeze(1) + self.step
        z = self.out(F.silu(base)) + self.skip(hn).unsqueeze(1) + self.res(F.silu(base))
        return z

    def config(self):
        return dict(d_model=self.d_model, n_draft=self.n_draft, expansion=self.expansion)
