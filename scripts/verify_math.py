"""NumPy Monte-Carlo check that the accept/reject rule is lossless (no GPU or torch needed).
Target = Markov chain P; drafter distributions Q are deliberately poor and context-free.
The first 3 emitted tokens must follow the exact chain distribution."""
import numpy as np

rng = np.random.default_rng(0)
V, N, TRIALS = 5, 3, 300_000
P = rng.dirichlet(np.ones(V) * 0.6, size=V)      # target: P[prev] = next-token distribution
Q = rng.dirichlet(np.ones(V) * 2.0, size=N)      # arbitrary drafter distribution per offset
pi0 = rng.dirichlet(np.ones(V))                  # distribution of the pending token y


def cycle(y):
    d = [rng.choice(V, p=Q[k]) for k in range(N)]
    rows = [P[y]] + [P[t] for t in d]            # p_0..p_N given the drafted prefix
    n = 0
    while n < N and rng.random() * Q[n][d[n]] < rows[n][d[n]]:
        n += 1
    res = rows[n] if n == N else np.maximum(rows[n] - Q[n], 0)
    return d[:n] + [rng.choice(V, p=res / res.sum())]


counts, cycles, emitted = np.zeros((V, V, V)), 0, 0
for _ in range(TRIALS):
    seq, prev = [], rng.choice(V, p=pi0)
    while len(seq) < 3:
        new = cycle(seq[-1] if seq else prev)
        cycles += 1; emitted += len(new); seq += new
    counts[seq[0], seq[1], seq[2]] += 1
emp = counts / TRIALS
exact = np.einsum("y,ya,ab,bc->abc", pi0, P, P, P)
print(f"max |empirical - exact| over {V**3} sequences: {np.abs(emp - exact).max():.5f}  (MC noise ~ {1/np.sqrt(TRIALS):.4f})")
print(f"total-variation distance: {0.5 * np.abs(emp - exact).sum():.5f}")
print(f"tokens emitted per verify cycle with a poor drafter: {emitted / cycles:.3f} (always >= 1)")
