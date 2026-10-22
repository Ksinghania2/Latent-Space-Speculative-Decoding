# Latent-Space Speculative Decoding for Sub-50 ms Edge Models

A **latent draft-and-verify engine**: a small head reads an *intermediate* hidden state of a frozen open-weights LLM and
predicts the final-layer latents of the next **N** tokens in a single pass. Those latents are decoded by the model's own
LM head into draft tokens, and the target model verifies all of them in **one** forward pass with exact
acceptance-rejection sampling, so the output distribution is identical to normal decoding.

* `latentspec/` core library (draft head, sampling, engine, HF backbone)
* `scripts/train_head.py` train the head · `scripts/benchmark.py` latency/acceptance/losslessness · `scripts/demo.py` terminal demo
* `scripts/verify_math.py` NumPy proof-by-simulation of losslessness (no GPU needed) · `scripts/roofline.py` bandwidth model
* `docs/index.html` **interactive lab** (GitHub Pages) · `REPORT.md` full technical report · `tests/` pytest

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_math.py                            # losslessness check, runs anywhere
python scripts/train_head.py --model Qwen/Qwen2.5-0.5B --n-draft 4 --steps 3000   # GPU recommended
python scripts/benchmark.py --head results/head.pt --max-new 16 --temp 0          # writes docs/trace.json
python scripts/demo.py --head results/head.pt
python -m http.server -d docs 8000                       # open http://localhost:8000
pytest -q
```

## Publish from VS Code
1. Open the folder in VS Code → Source Control → **Initialize Repository** (or `git init`).
2. Commit, then **Publish Branch** (VS Code creates the GitHub repo), or `git remote add origin <url> && git push -u origin main`.
3. GitHub → Settings → Pages → Source: *Deploy from a branch* → `main` / `/docs`. The lab goes live at `https://<user>.github.io/<repo>/`.
4. Commit your own `docs/trace.json` (from `benchmark.py`) and the page switches from simulator to **measured trace** automatically.

## Status
Verified in development: Python compiles, the accept/reject maths passes the Monte-Carlo test, JS syntax checks. **Not yet run:** the
PyTorch/HF path (training, benchmark) needs a machine with `torch` + model weights; no latency numbers are claimed until you run it.
