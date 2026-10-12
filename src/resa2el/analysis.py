"""Exact probabilities for small pools, to compare with simulation.

TODO (Rodrigo): write these after the samplers work.
"""
import numpy as np


def simulate_inclusion(sampler, scores, s_pos, sims, k, rng, repeats):
    """Estimate P(page i is selected) for each page by repeating the sampler."""
    raise NotImplementedError


def exact_stage1_inclusion(weights, n_keep):
    """Exact P(page i is kept) under successive sampling.
    Enumerate every ordered draw sequence. Use only for small pools (N <= 8)."""
    raise NotImplementedError
