"""Analytic memory-bandwidth roofline. Weights are read once per forward pass, so decoding is bandwidth-bound:
    t_pass >= bytes_of_weights / bandwidth.   Speculative cycle emits E[tau] tokens per pass."""
import argparse

def expected_tokens(alphas):
    e, prod = 1.0, 1.0
    for a in alphas:
        prod *= a; e += prod
    return e

ap = argparse.ArgumentParser(); ap.add_argument("--params-b", type=float, default=0.494)
ap.add_argument("--alpha1", type=float, default=0.8); ap.add_argument("--rho", type=float, default=0.85)
ap.add_argument("--n", type=int, default=4); ap.add_argument("--resp", type=int, default=16)
a = ap.parse_args()
DEV = {"Jetson Orin Nano (68 GB/s)": 68, "Jetson Orin NX 16GB (102 GB/s)": 102, "Apple M2 (100 GB/s)": 100,
       "RTX 4060 (272 GB/s)": 272, "RTX 4090 (1008 GB/s)": 1008}
tau = expected_tokens([a.alpha1 * a.rho ** k for k in range(a.n)])
print(f"E[tokens/cycle] = {tau:.2f}  (alpha1={a.alpha1}, rho={a.rho}, N={a.n})")
print(f"{'device':34}{'prec':>6}{'ms/pass':>9}{'AR '+str(a.resp)+'tok':>10}{'spec '+str(a.resp)+'tok':>11}")
for name, bw in DEV.items():
    for prec, bpp in (("fp16", 2), ("int8", 1), ("int4", 0.5)):
        t = a.params_b * 1e9 * bpp / (bw * 1e9) * 1e3
        print(f"{name:34}{prec:>6}{t:9.2f}{a.resp*t:10.1f}{(a.resp/tau)*t*1.05:11.1f}")
