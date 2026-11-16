"""Build the candidate pages of a training batch (numpy only, so it can be tested without a GPU).

Column 0 of the result is the correct page. Columns 1..k are the negatives.
"""
import numpy as np

from . import samplers as S


def split_heldout(keep, n_heldout, seed=12345):
    """Take n_heldout mentions out of `keep` (rows of the pool file). The choice
    depends only on `seed`, so every run holds out the same mentions.
    Returns (train rows, held-out rows)."""
    if not n_heldout:
        return keep, keep[:0]
    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(keep, n_heldout, replace=False))
    return keep[~np.isin(keep, held)], held


def batch_candidates(sample_idx, pools, sampler_name, k, rng, sampler_kw=None, gold_ids=None):
    """Page ids for each mention of the batch, shape (len(sample_idx), k + 1)."""
    ids, scores, gold, gold_score, sims = (pools[x] for x in ("ids", "scores", "gold", "gold_score", "sims"))
    cand = np.empty((len(sample_idx), k + 1), dtype=np.int64)
    for r, i in enumerate(sample_idx):
        cand[r, 0] = gold[i]
        if sampler_name == "gold_random":
            cand[r, 1:] = S.gold_random(gold_ids, gold[i], k, rng)
            continue
        not_gold = ids[i] != gold[i]  # the correct page is removed from the pool
        pool_ids = ids[i][not_gold]
        sel = S.SAMPLERS[sampler_name](scores[i][not_gold].astype(np.float64), float(gold_score[i]),
                                       sims[i][not_gold].astype(np.float64), k, rng, **(sampler_kw or {}))
        cand[r, 1:] = pool_ids[np.asarray(sel)]
    return cand


def live_select(live_scores, live_pos, name, k, rng, temp=1.0):
    """Negatives chosen from live scores. `live_scores` has shape (B, N): the score of
    each wrong pool page under the model in training. `live_pos` (B,) is the live score
    of the correct page. Returns (sel, corr), both of shape (B, k): the columns picked,
    and the number to subtract from the logit of each pick (zero unless score_prop)."""
    B = len(live_scores)
    sel = np.empty((B, k), dtype=np.int64)
    corr = np.zeros((B, k), dtype=np.float32)
    for r in range(B):
        sc = np.asarray(live_scores[r], dtype=np.float64)
        if name == "live_top":
            sel[r] = S.topscore(sc, float(live_pos[r]), None, k, rng)
        elif name == "live_top_below":
            sel[r] = S.top_below(sc, float(live_pos[r]), None, k, rng)
        elif name == "score_prop":
            sel[r], logq = S.score_prop(sc, k, rng, temp)
            corr[r] = np.log(k) + logq
        else:
            raise ValueError(name)
    return sel, corr
