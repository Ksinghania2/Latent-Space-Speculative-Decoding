# Latent-Space Speculative Decoding for Sub-50 ms Edge Models — Technical Report

## 1. Abstract
Autoregressive decoding of a billion-parameter-class LLM is memory-bandwidth bound: every token streams all weights from DRAM
while using a tiny fraction of available FLOPs. We build a **latent draft-and-verify engine** that (i) trains a small head on an
intermediate layer of a frozen model to predict the final-layer latents of the next *N* tokens in one pass, (ii) decodes them with the
model's own LM head into draft tokens, and (iii) verifies all drafts in a single target pass via acceptance-rejection sampling, which is
provably distribution-preserving. Drafting is free of extra model passes because the verification pass of cycle *t* exposes the hidden state
needed to draft cycle *t+1*. We provide the implementation, a NumPy proof-by-simulation of losslessness, an analytic roofline for edge devices,
and an interactive lab. **Measured GPU latency is produced by the included benchmark and is not reported here** (see §7).

## 2. Background: why decoding is bandwidth-bound
A forward pass for one token performs ≈ 2·P FLOPs but must read P·b bytes of weights (b = bytes/param). Arithmetic intensity ≈ 2/b FLOP/byte
(1 for fp16), far below the ridge point of any modern accelerator (tens to hundreds). Hence
`t_pass ≳ P·b / BW`. Verifying *k+1* tokens in one pass reads the same weights, so its cost is almost identical to one token (until the
pass becomes compute-bound at roughly 10²-token batches). Any scheme that emits more than one *correct* token per pass converts idle
FLOPs into latency savings.

## 3. Method
### 3.1 Cycle
State: committed prefix with KV cache of length *c*; pending token *y* (already sampled, not yet fed); drafts *d₁…d_N*.
1. **Verify pass.** Feed `[y, d₁..d_N]` (N+1 tokens) through the target. Row *k* of the final logits is the target distribution *p_k* for the token at draft slot *k+1*; row *N* is a bonus slot.
2. **Accept/reject.** For k=1..N accept d_k with probability min(1, p_{k-1}(d_k)/q_k(d_k)); at the first rejection draw the replacement from the residual `norm(max(p−q,0))`; if all pass, draw the bonus token from *p_N*. 
3. **Commit & roll back.** Emit the accepted drafts plus the new token; crop the KV cache to `c+1+n_acc`.
4. **Draft.** From the same pass take `h_mid` at the last kept position and the embedding of the new pending token; the head outputs N latents; the frozen LM head decodes them to q₁…q_N; sample drafts.

### 3.2 Latent draft head (`draft_head.py`)
Input `[RMSNorm(h_mid) ; RMSNorm(emb(y))]` → fuse linear (2d→2d) → SiLU/linear → add a learned per-offset vector → SiLU → output linear (→d), plus a direct linear skip from `h_mid`.
All N latents come from **one** pass; the offset embedding makes them distinct. Target: `z_k ≈ h_final[s+k]` (post-norm final hidden state at position s+k, k=1..N), where *s* is the last kept position.
For Qwen2.5-0.5B (d=896) the head has ≈ 9 M parameters (~1.8 % of the target), i.e. ~18 MB in fp16.

**Why latents, not token logits?** Predicting in the target's own representation space lets us reuse the frozen LM head as the decoder, so the draft
distribution lives in the same geometry as the target's and N logit rows cost one weight read. The token-space KL term keeps decoded drafts aligned with what matters (the ranking of vocabulary items).

### 3.3 Training objective
`L = MSE(z, h_final)/E[h_final²] + KL( softmax(W h_final) ‖ softmax(W z) )` with W the frozen LM head (fp32 copy). The backbone is never updated, so no behaviour drift
and training needs only forward passes of the target (data: WikiText-2 by default or any text file).

### 3.4 Correctness (losslessness)
Standard speculative-sampling argument: for a draft x~q, accepting with probability min(1,p/q) yields mass min(p,q) on x; the rejected mass `1−Σmin(p,q)` is exactly the normaliser of the residual `max(p−q,0)`, so
sampling the residual on rejection restores *p* exactly. Induction over positions (each p_k is conditioned on the already-accepted prefix) gives the target joint distribution. Drafts are sampled from q, so *q may be arbitrarily poor* — it only affects speed.
**Empirical check** (`scripts/verify_math.py`, 300 000 trials, target = random 5-state Markov chain, deliberately bad drafter, N=3):
max |empirical − exact| over 125 length-3 sequences = **0.00073** (Monte-Carlo noise ≈ 0.0018), total-variation distance 0.006; the drafter still yielded 2.12 tokens per cycle. At T=0 the rule reduces to "accept iff argmax p = draft", so greedy output must equal plain greedy decoding (the benchmark asserts this).

### 3.5 Expected speed-up
With per-offset acceptance α_k, tokens per cycle `E[τ] = 1 + Σ_{j=1..N} Π_{k≤j} α_k` and
`speed-up ≈ E[τ]·t_pass / (t_pass·(1+ε_N) + t_draft)`, where ε_N is the small extra cost of N more tokens in the pass. Example: α₁=0.8, ρ=0.85, N=4 ⇒ E[τ]=2.81.

## 4. Systems / execution-pipeline optimisations
| Technique | Where | Effect |
|---|---|---|
| Single pass yields verify logits, bonus token **and** next draft's input state | `engine.py` | removes the separate drafter pass |
| N latents decoded as one `(N,d)@(d,V)` matmul | `_draft` | one read of LM-head weights (often the largest single matrix on small models) |
| Branch-free vectorised accept/reject (`cumprod`, zero-padded q row, exponential-trick sampling) | `sampling.py` | no Python loop, no `multinomial` sync; one host sync per cycle (to know *n_acc*) |
| KV-cache crop instead of recompute | `Backbone.crop` | O(1) rollback of rejected drafts |
| fp16/bf16 weights, optional `torch.compile` of head and verifier (`--compile`) | `SpecEngine` | fuses elementwise ops around the small kernels |
| Draft head ≈ 2 % of target | — | drafting cost ≪ one pass |
**Not implemented (future work):** CUDA-graph capture with a static KV cache (removes Python/launch overhead, which dominates on small models), a fused
LM-head + softmax + accept kernel in Triton/CUDA, tree-shaped drafts, int4/int8 weights via bitsandbytes/AWQ. These are the main levers for the last 2-3× toward the 50 ms target on very small devices.

## 5. Roofline for edge devices (`scripts/roofline.py`)
Floor for a 0.494 B-parameter model, 16-token response, α₁=0.8, ρ=0.85, N=4 (E[τ]=2.81), 5 % cycle overhead. Nominal bandwidth specs; **real systems are slower** (launch overhead, KV reads, non-peak bandwidth):

| Device | Precision | ms/pass floor | AR 16 tok | Speculative 16 tok |
|---|---|---|---|---|
| Jetson Orin Nano (68 GB/s) | fp16 / int8 / int4 | 14.5 / 7.3 / 3.6 | 232 / 116 / 58 ms | 87 / 43 / 22 ms |
| Jetson Orin NX 16 GB (102 GB/s) | fp16 / int8 / int4 | 9.7 / 4.8 / 2.4 | 155 / 78 / 39 ms | 58 / 29 / 15 ms |
| RTX 4060 (272 GB/s) | fp16 / int8 / int4 | 3.6 / 1.8 / 0.9 | 58 / 29 / 15 ms | 22 / 11 / 5 ms |
**Reading:** speculation gives ~2.7× over the AR floor; sub-50 ms for a ~16-token response is *plausible* on 8-bit/4-bit or higher-bandwidth edge hardware once overheads are engineered away (CUDA graphs), and not expected for fp16 on the lowest-bandwidth boards. "Sub-50 ms" is thus a **response-length × hardware × precision** statement; we deliberately make no blanket claim.

## 6. Interactive lab (`docs/index.html`)
Race view of autoregressive vs latent-speculative decoding (green = accepted drafts, red = rejected, blue = bonus/corrected), live controls for N, α₁, ρ, pass time, overhead and response length,
a 50 ms budget gauge, and a speed-up-vs-N chart. In *simulator mode* outcomes are sampled from the parametric acceptance model; with `docs/trace.json` (from the benchmark) or a loaded trace file it replays **measured** cycles and timings.

## 7. Experimental protocol and what has / hasn't been run
**Validated in the current CPU environment:** the pytest suite; the NumPy losslessness simulation (§3.4); and an end-to-end greedy smoke test using the cached Qwen2.5-0.5B model and head. For a four-token request, speculative output exactly matched autoregressive output. This environment used Python 3.14.2, PyTorch 2.14.1+cpu, and Transformers 5.20.2.

**Saved benchmark result (`results/bench.json`):** greedy equality was true, but mean speculative latency was 17.09 s versus 9.19 s autoregressive (0.58× speedup) on CPU; first-offset acceptance was only 2.3%. This is evidence that the current head and CPU setup do not accelerate decoding. No fresh GPU benchmark or GPU training has been validated, and no sub-50 ms result is claimed.

**To reproduce and evaluate:** (1) `train_head.py` 3 000 steps (batch 8×256 tokens) on the intended hardware; (2) `benchmark.py --temp 0` ⇒ `lossless_greedy: true`; (3) run the benchmark across representative prompts and with `--temp 0.7`; (4) record `mean_speedup`, per-offset α, and response latency in `results/bench.json`. The cached CPU model smoke test does not replace this protocol.
**Suggested ablations:** mid-layer choice (⅓, ½, ⅔ depth); N ∈ {2,4,6,8}; MSE-only vs KL-only vs both; head width; training-set domain vs eval domain; T ∈ {0, 0.7, 1.0}.
**Success criteria:** greedy equality with baseline; mean tokens/cycle ≥ 2; wall-clock speed-up ≥ 1.8× at N=4 on a GPU; ms/response vs the 50 ms budget at your chosen length/precision.

## 8. Limitations and risks
* Acceptance depends on how predictable text is; open-ended creative text accepts less than code or boilerplate. Latent errors compound with offset (hence ρ<1).
* Independent per-offset drafts (a chain-structured head or tree verification would improve α at higher N).
* Verification compute grows with N+1 tokens; on weak-compute edge chips the benefit saturates earlier.
* Top-k/top-p sampling is supported by the sampling primitives and must be applied identically to target and draft distributions to preserve the speculative-sampling guarantee.
* Batch size 1 only; the code targets single-user edge inference.
* Hugging Face `DynamicCache.crop` API and `hidden_states[-1]` being post-norm hold for Llama/Qwen/GPT-2 families; the trainer prints a sanity check.

## 9. Related work
Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (2023); Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023);
Cai et al., *Medusa* (2024) – multiple decoding heads; Li et al., *EAGLE* (2024) – feature-level (hidden-state) drafting, the closest relative of our latent approach; Elhoushi et al., *LayerSkip* (2024) – early-exit self-speculation.
Our contribution here is an integrated, minimal, reproducible implementation emphasising single-pass latent drafting, a sync-light vectorised verifier, and an edge roofline analysis.

## 10. Conclusion
Latent drafting with exact verification trades idle FLOPs for fewer weight reads. The maths is verified lossless; the speed-up and sub-50 ms outcome depend on measured acceptance and device bandwidth, for which the repo provides the full train-benchmark-visualise pipeline.
