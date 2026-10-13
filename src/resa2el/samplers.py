"""Negative samplers for one query.

Every sampler has the same inputs and output, so the training code can swap them.

Inputs
  scores  shape (N,)  retriever score of each candidate page (correct page removed)
  s_pos   float       retriever score of the correct page
  sims    shape (N,)  similarity of each candidate page to the correct page
  k       int         number of negatives to return
  rng     numpy.random.Generator

Output
  array of k distinct indices into `scores`

Sizes from the ReSa2 code (github.com/ad32q/ReSa2):
  pool 1,000 -> stage 1 keeps 500 -> stage 2 ranks, keeps top 75 -> select 7.

TODO (Rodrigo): write these functions. The tests in tests/test_samplers.py
say what each one must do. sim/toy_pool.py has an older model for reference.
"""
import numpy as np

TAU = 0.25  # stage 1 weight width, same value as firstSampling.py


def uniform(scores, s_pos, sims, k, rng):
    """Select k pages uniformly, without replacement."""
    return rng.choice(len(scores), k, replace=False)


def topscore(scores, s_pos, sims, k, rng):
    """Select the k pages with the highest score. No randomness."""
    return np.argsort(scores)[-k:]


def stage1_weights(scores, s_pos, tau=TAU):
    """Weight of each page: exp(-tau * (score - s_pos)^2)."""
    raise NotImplementedError


def resa2_stage1(scores, s_pos, n_keep, rng, tau=TAU):
    """Keep n_keep pages. Draw one page with prob. proportional to its weight,
    remove it, and draw again from the rest (successive sampling)."""
    raise NotImplementedError


def resa2(scores, s_pos, sims, k, rng, n_stage1=500, n_stage2=75, tau=TAU):
    """Stage 1, then rank the kept pages by `sims`, keep the top n_stage2,
    and select k of them uniformly."""
    raise NotImplementedError


SAMPLERS = {"uniform": uniform, "topscore": topscore, "resa2": resa2}
