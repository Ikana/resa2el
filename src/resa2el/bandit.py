"""EXP3 bandit over a small number of arms (Auer et al. 2002), with rewards scaled by percentiles.

One round = one interval of training with one arm (a similarity band).
The raw reward is the fall of the held-out loss in the interval. It is clipped
to the 20th and 80th percentiles of the rewards seen so far and mapped to
0 to 1 (the scale of Graves et al. 2017, moved from [-1, 1] to [0, 1]).
"""
import numpy as np


class Exp3:
    def __init__(self, n_arms, n_rounds, gamma=None, eta=None, lo_pct=20, hi_pct=80, min_history=5):
        self.k = n_arms
        if gamma is None:  # Auer et al. 2002, Corollary 3.2 with the trivial bound on the best gain
            gamma = min(1.0, float(np.sqrt(n_arms * np.log(n_arms) / ((np.e - 1) * n_rounds))))
        self.gamma = gamma
        self.eta = gamma / n_arms if eta is None else eta
        self.w = np.zeros(n_arms)  # log weights
        self.history = []
        self.lo_pct, self.hi_pct, self.min_history = lo_pct, hi_pct, min_history

    def probs(self):
        p = np.exp(self.w - self.w.max())
        return (1 - self.gamma) * p / p.sum() + self.gamma / self.k

    def pick(self, rng):
        return int(rng.choice(self.k, p=self.probs()))

    def scale(self, raw):
        """Map a raw reward to 0 to 1 with the percentiles of the past rewards."""
        if len(self.history) < self.min_history:
            return 0.5
        lo, hi = np.percentile(self.history, [self.lo_pct, self.hi_pct])
        if hi <= lo:
            return 0.5
        return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))

    def update(self, arm, raw):
        """Give the raw reward of the arm that was picked. Returns the scaled reward."""
        r = self.scale(raw)
        self.history.append(float(raw))
        self.w[arm] += self.eta * r / self.probs()[arm]
        return r
