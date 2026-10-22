"""Train the latent draft head on a FROZEN backbone.
Loss = normalised MSE in latent space + KL(p_target || softmax(lm_head(z))) in token space.
    python scripts/train_head.py --model Qwen/Qwen2.5-0.5B --n-draft 4 --steps 3000
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, math, torch, torch.nn.functional as F
from latentspec import LatentDraftHead
from latentspec.backbone import Backbone


def load_tokens(bb, path):
    if path:
        text = open(path, encoding="utf-8").read()
    else:
        try:
            from datasets import load_dataset
            text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
        except Exception as e:
            print("datasets unavailable or download failed:", e)
            # Fall back to a small built-in corpus so training can run offline.
            text = ("The quick brown fox jumps over the lazy dog. " * 1000 +
                    "In a hole in the ground there lived a hobbit. " * 1000)
    return bb.tok(text, return_tensors="pt").input_ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--text", default=None)
    ap.add_argument("--n-draft", type=int, default=4); ap.add_argument("--mid-layer", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256); ap.add_argument("--pos", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--out", default="results/head.pt")
    a = ap.parse_args()
    bb = Backbone(a.model, mid_layer=a.mid_layer); dev, N, d = bb.device, a.n_draft, bb.d
    toks = load_tokens(bb, a.text); print(f"{len(toks):,} training tokens, mid layer {bb.mid}/{bb.n_layers}")
    head = LatentDraftHead(d, N).to(dev).float()
    W = bb.lm_head.weight.float()                                  # frozen decoder, fp32 copy
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    ar = torch.arange(a.batch, device=dev)[:, None]
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        s = torch.randint(0, len(toks) - a.seq - 1, (a.batch,))
        batch = torch.stack([toks[i:i + a.seq] for i in s]).to(dev)
        with torch.no_grad():
            out = bb.model(input_ids=batch, output_hidden_states=True)
            hm, hf = out.hidden_states[bb.mid].float(), out.hidden_states[-1].float()
            if step == 0:
                print("sanity |lm_head(h_final)-logits| max:",
                      float((hf[:, -1] @ W.T - out.logits[:, -1].float()).abs().max()))
        pos = torch.randint(0, a.seq - N - 1, (a.batch, a.pos), device=dev)
        e = bb.embed(batch[ar, pos + 1]).float()
        tgt = torch.stack([hf[ar, pos + k] for k in range(1, N + 1)], 2)           # (B,K,N,d)
        z = head(hm[ar, pos].reshape(-1, d), e.reshape(-1, d)).view(a.batch, a.pos, N, d)
        mse = F.mse_loss(z, tgt) / tgt.pow(2).mean()
        lq = F.log_softmax(z @ W.T, -1)
        with torch.no_grad():
            pt = F.softmax(tgt @ W.T, -1)
        kl = F.kl_div(lq.flatten(0, 2), pt.flatten(0, 2), reduction="batchmean")
        loss = mse + kl
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            top1 = (z @ W.T).argmax(-1).eq((tgt @ W.T).argmax(-1)).float().mean(dim=(0, 1))
            print(f"step {step:5d} loss {loss:.3f} mse {mse:.3f} kl {kl:.3f} top1/offset {[round(float(t), 2) for t in top1]}")
    torch.save(dict(state=head.state_dict(), cfg=head.config(), model=a.model, mid=bb.mid), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
