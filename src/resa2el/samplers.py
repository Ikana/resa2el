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

The tests in tests/test_samplers.py say what each one must do.
sim/toy_pool.py has an older model for reference.
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
    return np.exp(-tau * (scores - s_pos) ** 2)


def resa2_stage1(scores, s_pos, n_keep, rng, tau=TAU):
    """Keep n_keep pages. Draw one page with prob. proportional to its weight,
    remove it, and draw again from the rest (successive sampling)."""
    w = stage1_weights(scores, s_pos, tau)
    n_keep = min(n_keep, len(scores))
    # numpy draws without replacement one at a time, with the weights of the
    # pages that are left: this is successive sampling.
    # A tiny floor keeps every page possible, so n_keep draws always exist.
    w = w + 1e-300
    return rng.choice(len(scores), n_keep, replace=False, p=w / w.sum())


def resa2(scores, s_pos, sims, k, rng, n_stage1=500, n_stage2=75, tau=TAU):
    """Stage 1, then rank the kept pages by `sims`, keep the top n_stage2,
    and select k of them uniformly."""
    kept = resa2_stage1(scores, s_pos, n_stage1, rng, tau)
    order = np.argsort(-np.asarray(sims)[kept], kind="stable")
    top = kept[order[:n_stage2]]
    return rng.choice(top, min(k, len(top)), replace=False)


# Similarity bands used in the paper (ranks by similarity to the correct page, 0 = most similar).
BANDS = {"top25": (0, 25), "25to75": (25, 75), "75to300": (75, 300), "rest": (300, None)}


def band(scores, s_pos, sims, k, rng, lo=0, hi=75):
    """Select k pages uniformly from one similarity band: the pages whose
    similarity rank r satisfies lo <= r < hi (hi=None: to the end of the pool)."""
    order = np.argsort(-np.asarray(sims), kind="stable")
    part = order[lo:hi]
    return rng.choice(part, min(k, len(part)), replace=False)


def stage2_only(scores, s_pos, sims, k, rng, n_stage2=75):
    """ReSa2 without stage 1: k uniform picks from the n_stage2 pool pages most similar to the correct page."""
    return band(scores, s_pos, sims, k, rng, lo=0, hi=n_stage2)


def allpool(scores, s_pos, sims, k, rng):
    """No sampling: the k highest-scoring pages, with k set to the pool size
    (999, the smallest number of wrong pages a pool has). No randomness."""
    return np.argsort(scores)[-k:]


def gold_random(gold_of_mentions, own_gold, k, rng):
    """k distinct correct pages of other training mentions. `gold_of_mentions` has
    one entry per training mention, so a page that is the answer of many mentions
    is drawn more often, as it is inside a training batch. Returns page ids, not
    pool indices. This is the in-batch idea with stored page vectors."""
    out = []
    seen = {int(own_gold)}
    while len(out) < k:
        for g in rng.choice(gold_of_mentions, 2 * k).tolist():
            if g not in seen:
                seen.add(g); out.append(g)
                if len(out) == k:
                    break
    return np.asarray(out)


SAMPLERS = {"uniform": uniform, "topscore": topscore, "resa2": resa2,
            "band": band, "stage2_only": stage2_only, "allpool": allpool}
