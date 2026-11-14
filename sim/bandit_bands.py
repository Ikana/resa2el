"""Prototype for the bandit sampler (wayfinder ticket ROD-129). CPU only, no data.

Question: can EXP3 find the best similarity band in 1,500 rounds?

Decided so far:
  arms    4 similarity bands (top 25, 25-75, 75-300, the rest)
  reward  learning progress, scaled to [0, 1], one reward per update
  rule    EXP3 (Auer et al. 2002), one bandit for all mentions

We do not know yet how different the bands are. So this script does not
simulate the retriever. It gives each band a mean reward, adds noise, and
asks: how large must the difference between bands be before EXP3 sees it?

  delta   mean reward of the best band minus the mean reward of the others
  noise   standard deviation of one reward (rewards are clipped to [0, 1])

Run:  python3 sim/bandit_bands.py
"""
import numpy as np

K = 4            # bands
T = 1500         # rounds = updates
REPEATS = 300
NOISE = 0.25
BASE = 0.5       # mean reward of an ordinary band


def exp3(means_at, gamma, eta, rng):
    """One EXP3 run. means_at(t) -> mean reward of each band at round t.
    Returns the probability vector at every round."""
    w = np.zeros(K)                      # log weights
    probs = np.empty((T, K))
    for t in range(T):
        p = np.exp(w - w.max())
        p = (1 - gamma) * p / p.sum() + gamma / K
        probs[t] = p
        arm = rng.choice(K, p=p)
        r = float(np.clip(rng.normal(means_at(t)[arm], NOISE), 0, 1))
        w[arm] += eta * r / p[arm]       # importance-weighted reward
    return probs


def fixed_best(delta):
    m = np.full(K, BASE); m[1] += delta          # band 2 (25-75) is best
    return lambda t: m


def switching(delta):
    a = np.full(K, BASE); a[3] += delta          # first half: the easy band
    b = np.full(K, BASE); b[1] += delta          # second half: band 2
    return lambda t: a if t < T // 2 else b


def summarize(make_means, best_late, gamma, eta):
    rows = []
    for delta in (0.02, 0.05, 0.10, 0.20):
        rng = np.random.default_rng(0)
        runs = np.stack([exp3(make_means(delta), gamma, eta, rng) for _ in range(REPEATS)])
        last = runs[:, -150:, best_late].mean()          # last 10% of training
        whole = runs[:, :, best_late].mean()
        rows.append((delta, whole, last))
    return rows


def main():
    # Tuning from Auer et al. 2002 (Corollary 3.2 form): gamma = sqrt(K ln K / ((e-1) T))
    g_theory = min(1.0, np.sqrt(K * np.log(K) / ((np.e - 1) * T)))
    settings = [("theory", g_theory, g_theory / K),
                ("4x faster", 4 * g_theory, 4 * g_theory / K),
                ("10x faster", 0.25, 10 * g_theory / K)]
    print("Share of picks that go to the best band (uniform would be 0.25).")
    print("noise sd = %.2f, %d rounds, %d repeats\n" % (NOISE, T, REPEATS))
    for title, make, best in (("Best band never changes", fixed_best, 1),
                              ("Best band changes at half time", switching, 1)):
        print("##", title)
        print("| tuning | gamma | delta | whole run | last 150 rounds |")
        print("|---|---|---|---|---|")
        for name, gamma, eta in settings:
            for delta, whole, last in summarize(make, best, gamma, eta):
                print("| %s | %.3f | %.2f | %.2f | %.2f |" % (name, gamma, delta, whole, last))
        print()


if __name__ == "__main__":
    main()
