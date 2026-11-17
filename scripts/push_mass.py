"""Share of the pool's push that the 7 negatives of each sampler carry (added for resa2el).

For one mention, the weight of pool page i is w_i = exp(s_i - s_pos). The mass of a
selection S is M(S) = sum of w_i over S. The loss on S is log(1 + M(S)) and the total
push on the mention vector is M(S) / (1 + M(S)) (paper, section on the math).
Scores are those of the released model on stored pools, so this is the state before
the first update. Run: PYTHONPATH=src python scripts/push_mass.py POOLS.npz
"""
import json
import sys

import numpy as np

from resa2el.samplers import resa2, stage2_only, topscore, uniform

K, DRAWS = 7, 50


def main():
    z = np.load(sys.argv[1])
    rng = np.random.default_rng(0)
    rules = {"Uniform": uniform, "Stage 2 only": stage2_only, "ReSa2": resa2, "Top 7 by score": topscore}
    acc = {n: {"share": [], "loss": [], "push": []} for n in rules}
    full = {"loss": [], "push": []}
    for q in range(len(z["gold"])):
        s, sp, sims = z["scores"][q].astype(float), float(z["gold_score"][q]), z["sims"][q].astype(float)
        keep = z["ids"][q] != z["gold"][q]
        s, sims = s[keep], sims[keep]
        w = np.exp(s - sp)
        m_all = w.sum()
        full["loss"].append(np.log1p(m_all)); full["push"].append(m_all / (1 + m_all))
        for n, f in rules.items():
            ms = np.array([w[f(s, sp, sims, K, rng)].sum() for _ in range(1 if n.startswith("Top") else DRAWS)])
            acc[n]["share"].append((ms / m_all).mean())
            acc[n]["loss"].append(np.log1p(ms).mean())
            acc[n]["push"].append((ms / (1 + ms)).mean())
    out = {"mentions": int(len(z["gold"])), "all_pool": {k: float(np.mean(v)) for k, v in full.items()}}
    print("| Rule | Share of the pool mass | Loss | Total push |\n|---|---|---|---|")
    for n in rules:
        r = {k: float(np.mean(v)) for k, v in acc[n].items()}
        r["share_median"] = float(np.median(acc[n]["share"]))
        out[n] = r
        print("| {} | {:.1f}% (median {:.1f}%) | {:.2f} | {:.2f} |".format(n, 100 * r["share"], 100 * r["share_median"], r["loss"], r["push"]))
    print("| All pool pages | 100% | {:.2f} | {:.2f} |".format(out["all_pool"]["loss"], out["all_pool"]["push"]))
    json.dump(out, open("docs/results/plan/push_mass.json", "w"), indent=1)


if __name__ == "__main__":
    main()
