"""Small model of the negative samplers in the proposal.

No BERT, no data. One query, 1,000 fake candidate pages with fake scores.
We repeat each sampler many times and count how often each page is selected.

Samplers (sizes from the ReSa2 code, github.com/ad32q/ReSa2):
  resa2    stage 1: sample 500 of the top 1,000 without replacement,
                    weight w = exp(-TAU * (s - s_pos)^2)       (firstSampling.py)
           stage 2: rank the 500 by similarity to the correct page,
                    keep the top 75, select 7 uniformly       (secondRetrieve.py)
  uniform  select 7 uniformly from the 1,000
  topscore select the 7 highest-scoring pages

Run:  python3 sim/toy_pool.py            (change the settings below and run again)
"""
import numpy as np

# ---------------- settings ----------------
N_POOL = 1000      # candidates retrieved for the query
N_STAGE1 = 500     # kept by stage 1
N_STAGE2 = 75      # kept by stage 2
N_NEG = 7          # negatives given to training
TAU = 0.25         # same value as firstSampling.py
REPEATS = 5000
SEED = 0

S_POS = 0.0        # score of the correct page
NEG_MEAN, NEG_SD = -4.0, 2.5   # scores of ordinary wrong pages

# Planted pages: (label, how many, score minus S_POS, similarity to correct page)
PLANTED = [
    ("wrong_year_close", 3, +0.5, 0.95),  # e.g. "2001 NFL Draft", scores a bit above correct
    ("wrong_year_far",   3, +4.0, 0.95),  # same, but scores far above correct
    ("parent_or_sub",    3, +0.3, 0.85),  # e.g. "National Football League Draft"
]
# ------------------------------------------


def make_pool(rng):
    """Return scores, similarity to the correct page, and a label for each candidate."""
    n_planted = sum(n for _, n, _, _ in PLANTED)
    n_ord = N_POOL - n_planted
    scores = list(rng.normal(NEG_MEAN, NEG_SD, n_ord))
    # Ordinary pages: similarity to the correct page goes up a little with score.
    sims = list(np.clip(0.5 + 0.03 * (np.array(scores) - NEG_MEAN)
                        + rng.normal(0, 0.08, n_ord), 0, 0.9))
    labels = ["ordinary"] * n_ord
    for label, n, gap, sim in PLANTED:
        scores += [S_POS + gap] * n
        sims += [sim] * n
        labels += [label] * n
    return np.array(scores), np.array(sims), np.array(labels)


def stage1_weights(scores):
    return np.exp(-TAU * (scores - S_POS) ** 2)


def resa2(rng, scores, sims):
    w = stage1_weights(scores)
    # np.random.choice(replace=False, p=...) is what firstSampling.py uses:
    # draw one page with prob. w / sum(w), remove it, renormalize, repeat.
    kept = rng.choice(N_POOL, size=N_STAGE1, replace=False, p=w / w.sum())
    top = kept[np.argsort(-sims[kept])[:N_STAGE2]]
    return rng.choice(top, size=N_NEG, replace=False)


def uniform(rng, scores, sims):
    return rng.choice(N_POOL, size=N_NEG, replace=False)


def topscore(rng, scores, sims):
    return np.argsort(-scores)[:N_NEG]


SAMPLERS = {"resa2": resa2, "uniform": uniform, "topscore": topscore}


def run():
    rng = np.random.default_rng(SEED)
    scores, sims, labels = make_pool(rng)   # pool and scores stay fixed
    above = scores > S_POS                  # "possible false negatives"
    print(f"pool: {N_POOL} pages, {above.sum()} score above the correct page")
    print(f"repeats per sampler: {REPEATS}\n")

    results = {}
    for name, f in SAMPLERS.items():
        counts = np.zeros(N_POOL)
        n_above = []
        for _ in range(REPEATS):
            sel = f(rng, scores, sims)
            counts[sel] += 1
            n_above.append(above[sel].sum())
        results[name] = (counts / REPEATS, np.array(n_above))

    # 1. Inclusion probability per page type (mean over pages of that type)
    types = ["ordinary"] + [label for label, _, _, _ in PLANTED]
    print("P(page is selected), mean per page type")
    print(f"{'type':18s}" + "".join(f"{n:>10s}" for n in SAMPLERS))
    for t in types:
        m = labels == t
        print(f"{t:18s}" + "".join(f"{results[n][0][m].mean():10.4f}" for n in SAMPLERS))

    # 2. Coverage, false negatives, and variation between runs
    print("\nper sampler")
    print(f"{'':28s}" + "".join(f"{n:>10s}" for n in SAMPLERS))
    rows = [
        ("pages ever selected", lambda p, a: f"{(p > 0).sum():10d}"),
        ("E[# above correct] in 7", lambda p, a: f"{a.mean():10.3f}"),
        ("SD[# above correct]", lambda p, a: f"{a.std():10.3f}"),
    ]
    for title, fmt in rows:
        print(f"{title:28s}" + "".join(fmt(*results[n]) for n in SAMPLERS))

    # 3. Stage 1 alone: inclusion is NOT proportional to the weight
    rng1 = np.random.default_rng(SEED + 1)
    w = stage1_weights(scores)
    inc = np.zeros(N_POOL)
    for _ in range(2000):
        inc[rng1.choice(N_POOL, size=N_STAGE1, replace=False, p=w / w.sum())] += 1
    inc /= 2000
    naive = np.minimum(1, N_STAGE1 * w / w.sum())
    print("\nstage 1 only: weight vs P(kept in the 500)")
    print(f"{'score - s_pos':>14s}{'weight':>10s}{'500*w/sum':>11s}{'P(kept)':>10s}")
    for i in np.argsort(-scores)[::100]:
        print(f"{scores[i] - S_POS:14.2f}{w[i]:10.4f}{naive[i]:11.4f}{inc[i]:10.4f}")


if __name__ == "__main__":
    run()
