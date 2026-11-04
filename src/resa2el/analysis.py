"""Exact probabilities for small pools, to compare with simulation."""
import itertools

import numpy as np


def simulate_inclusion(sampler, scores, s_pos, sims, k, rng, repeats):
    """Estimate P(page i is selected) for each page by repeating the sampler."""
    counts = np.zeros(len(scores))
    for _ in range(repeats):
        counts[np.asarray(sampler(scores, s_pos, sims, k, rng))] += 1
    return counts / repeats


def exact_stage1_inclusion(weights, n_keep):
    """Exact P(page i is kept) under successive sampling.
    Enumerate every ordered draw sequence. Use only for small pools (N <= 8)."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    assert n <= 8, "too many pages to enumerate"
    prob = np.zeros(n)
    for seq in itertools.permutations(range(n), n_keep):
        p, left = 1.0, w.sum()
        for i in seq:
            p *= w[i] / left
            left -= w[i]
        prob[list(seq)] += p
    return prob
