"""Question 1: how each sampler changes the inclusion probability of each
candidate, the coverage, and the variation between runs. CPU only.

Run: python scripts/analyze_pools.py POOLS.npz [--mentions 500] [--repeats 200]
Writes docs/results/analysis/question1.md and question1.json.

For each sampled mention and each sampler, the sampler runs `repeats` times
with different seeds. From the selections we compute:
  inclusion  P(page i selected), per page; summarized by similarity to the
             correct page (high-sim pages are the "wrong year" look-alikes)
  coverage   number of distinct pool pages ever selected, out of 1,000
  variation  mean Jaccard overlap of the selected sets between two runs
  hardness   mean score of the negatives relative to the correct page
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from resa2el import samplers as S  # noqa: E402

K = 7


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools")
    ap.add_argument("--mentions", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", default="docs/results/analysis")
    args = ap.parse_args()
    d = np.load(args.pools)
    ok = np.flatnonzero(d["gold"] >= 0)
    rng0 = np.random.default_rng(0)
    picks = rng0.choice(ok, args.mentions, replace=False)
    names = ["uniform", "topscore", "resa2"]
    res = {n: {"incl_by_sim": np.zeros(4), "cov": [], "jac": [], "gap": [], "sim": []} for n in names}
    rank_bins = [0, 25, 75, 300, 10**9]  # similarity ranks: top 25, 25-75, 75-300, rest
    for i in picks:
        not_gold = d["ids"][i] != d["gold"][i]
        scores = d["scores"][i][not_gold].astype(np.float64)
        sims = d["sims"][i][not_gold].astype(np.float64)
        s_pos = float(d["gold_score"][i])
        n = len(scores)
        # similarity rank in the pool: 0 = most similar to the correct page
        sim_rank = np.empty(n, dtype=int)
        sim_rank[np.argsort(-sims, kind="stable")] = np.arange(n)
        for name in names:
            fn = S.SAMPLERS[name]
            sels = [np.asarray(fn(scores, s_pos, sims, K, np.random.default_rng(r))) for r in range(args.repeats)]
            counts = np.zeros(n)
            for s in sels:
                counts[s] += 1
            incl = counts / args.repeats
            b = np.digitize(sim_rank, rank_bins) - 1
            res[name]["incl_by_sim"] += np.array([incl[b == j].sum() for j in range(4)])
            res[name]["cov"].append(int((counts > 0).sum()))
            res[name]["jac"].append(float(np.mean([jaccard(sels[r], sels[r + 1]) for r in range(0, args.repeats - 1, 2)])))
            res[name]["gap"].append(float(np.mean([(scores[s] - s_pos).mean() for s in sels])))
            res[name]["sim"].append(float(np.mean([sim_rank[s].mean() for s in sels])))
    out = {}
    lines = ["# Question 1: inclusion probability, coverage, and variation between runs", "",
             "Validation pools, {} random mentions, {} runs of each sampler per mention, k = {} negatives.".format(args.mentions, args.repeats, K), "",
             "| Sampler | expected negatives per run by similarity rank to the correct page: top 25 / 25-75 / 75-300 / 300-1000 | distinct pages selected (of 1,000) | overlap of two runs (Jaccard) | mean score gap to the correct page | mean similarity rank |",
             "|---|---|---|---|---|---|"]
    for name in names:
        r = res[name]
        incl = r["incl_by_sim"] / args.mentions
        out[name] = {"expected_negatives_by_sim_bin": incl.tolist(), "coverage_mean": float(np.mean(r["cov"])),
                     "jaccard_mean": float(np.mean(r["jac"])), "score_gap_mean": float(np.mean(r["gap"])), "sim_mean": float(np.mean(r["sim"]))}
        lines.append("| %s | %.2f / %.2f / %.2f / %.2f | %.0f | %.2f | %+.1f | %.0f |" % (
            name, *incl, out[name]["coverage_mean"], out[name]["jaccard_mean"], out[name]["score_gap_mean"], out[name]["sim_mean"]))
    lines += ["", "How to read it:",
              "- A row sums to 7 (the 7 negatives of one run), split by the negative's similarity rank to the correct page (rank 0 = the most similar page in the pool, often a 'wrong year' look-alike).",
              "- Coverage: how many different pool pages a sampler ever selects over %d runs (uniform: close to all; topscore: exactly 7)." % args.repeats,
              "- Jaccard: 1.0 means two runs select the same 7 pages (no randomness); 0 means no overlap.",
              "- Score gap: negative means the negatives score below the correct page (easy); positive means above (hard)."]
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "question1.md"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(args.out, "question1.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
