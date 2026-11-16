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


def mixed(scores, s_pos, sims, k, rng, n_hard=4, n_near=25):
    """Hard and random together (the recipe of DPR and RocketQA): n_hard uniform picks
    from the n_near pool pages most similar to the correct page, and k - n_hard uniform
    picks from the other pool pages."""
    order = np.argsort(-np.asarray(sims), kind="stable")
    near, far = order[:n_near], order[n_near:]
    n_hard = min(n_hard, k, len(near))
    return np.concatenate([rng.choice(near, n_hard, replace=False),
                           rng.choice(far, k - n_hard, replace=False)])


def top_below(scores, s_pos, sims, k, rng):
    """The k highest-scoring pages among the pages that score below the correct page.
    A page that scores above the correct page is often a second correct answer or a
    near copy (a false negative), so it is skipped. If fewer than k pages are below,
    the pages just above the correct page fill the rest. No randomness."""
    order = np.argsort(-np.asarray(scores), kind="stable")  # highest score first
    n_above = int((np.asarray(scores) >= s_pos).sum())
    below, above = order[n_above:], order[:n_above]
    sel = below[:k]
    if len(sel) < k:
        sel = np.concatenate([sel, above[::-1][:k - len(sel)]])
    return sel


def score_prop(scores, k, rng, temp=1.0):
    """Sampled softmax with a correction. Draw k pages WITH replacement, page i with
    probability q_i = softmax(scores / temp)_i. Returns (indices, log q of each pick).

    The training loss subtracts log(k * q_i) from the logit of each pick. Then
    (1/k) * sum_i exp(s_i) / q_i is an unbiased estimate of the sum of exp(s_j) over
    all wrong pages of the pool, so the loss with k picks estimates the loss with the
    whole pool. With temp = 1 every pick gives the exact sum."""
    z = np.asarray(scores, dtype=np.float64) / temp
    z = z - z.max()
    logq = z - np.log(np.exp(z).sum())
    idx = rng.choice(len(z), k, replace=True, p=np.exp(logq))
    return idx, logq[idx]


SAMPLERS = {"uniform": uniform, "topscore": topscore, "resa2": resa2,
            "band": band, "stage2_only": stage2_only, "allpool": allpool, "mixed": mixed}

# Samplers that use the scores of the model in training (live scores), not the stored scores.
LIVE = ("live_top", "live_top_below", "score_prop")
