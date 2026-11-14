"""Bootstrap confidence intervals from the per-mention ranks, and what ReSa2 stage 1 keeps (added for resa2el). CPU only.

Run: python scripts/bootstrap_ranks.py [RANKS_DIR] --out docs/results/bootstrap
  RANKS_DIR  folder with released.npz and <sampler>_s<seed>.npz from scripts/eval_ranks.py.
             Without it, the script reads docs/results/gaps/ranks_compact.npz
             (the same ranks without the titles). With it, the script writes that file first.

Method: for each mention the hit (0 or 1) is averaged over the seeds of a
sampler first. Then the mentions are drawn again with replacement B times, and
the difference of the two means is computed for each draw. The interval is the
2.5 to 97.5 percentile of these differences.

Stage 1 table: on the 200 stored validation pools (docs/results/analysis/pools_sample200.npz),
how much of what stage 1 keeps is fixed.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from resa2el import samplers as S  # noqa: E402

COMPACT = os.path.join(ROOT, "docs", "results", "gaps", "ranks_compact.npz")
METRICS = [("top-1", 1), ("recall at 10", 10), ("recall at 100", 100)]


def load(ranks_dir):
    if ranks_dir:
        out = {}
        for f in sorted(glob.glob(os.path.join(ranks_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            name = os.path.basename(f)[:-4]
            out["rank_" + name] = d["rank"].astype(np.int16)
            out["split"], out["gold"] = d["split"], d["gold"]
        np.savez_compressed(COMPACT, **out)
    d = np.load(COMPACT)
    return d["split"], d["gold"], {k[5:]: d[k].astype(int) for k in d.files if k.startswith("rank_")}


def hits(ranks, names, sel, k):
    """Per-mention hit rate at k, averaged over the given runs."""
    return np.mean([((ranks[n][sel] >= 0) & (ranks[n][sel] < k)) for n in names], axis=0)


def boot(a, b, rng, B):
    diff = (a - b) * 100
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stage1_table(B=20):
    d = np.load(os.path.join(ROOT, "docs", "results", "analysis", "pools_sample200.npz"))
    rng = np.random.default_rng(0)
    closest, overlap, same75, top75 = [], [], [], []
    for i in range(len(d["gold"])):
        keep = d["ids"][i] != d["gold"][i]
        sc, sims, sp = d["scores"][i][keep].astype(np.float64), d["sims"][i][keep].astype(np.float64), float(d["gold_score"][i])
        near = set(np.argsort(np.abs(sc - sp))[:500].tolist())
        best75 = set(np.argsort(-sims, kind="stable")[:75].tolist())
        prev = prev75 = None
        for _ in range(B):
            kept = S.resa2_stage1(sc, sp, 500, rng)
            ks = set(kept.tolist())
            k75 = set(kept[np.argsort(-sims[kept], kind="stable")[:75]].tolist())
            closest.append(len(ks & near) / 500)
            top75.append(len(k75 & best75) / 75)
            if prev is not None:
                overlap.append(len(ks & prev) / 500); same75.append(len(k75 & prev75) / 75)
            prev, prev75 = ks, k75
    return {"kept_pages_among_500_closest_in_score": float(np.mean(closest)), "kept_pages_shared_by_two_draws": float(np.mean(overlap)),
            "stage2_pages_among_75_most_similar_of_pool": float(np.mean(top75)), "stage2_pages_shared_by_two_draws": float(np.mean(same75))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks_dir", nargs="?")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "results", "bootstrap"))
    ap.add_argument("--B", type=int, default=5000)
    args = ap.parse_args()
    split, gold, ranks = load(args.ranks_dir)
    rng = np.random.default_rng(0)
    seeds = lambda s: sorted(n for n in ranks if n.startswith(s + "_s"))  # noqa: E731
    val, nyt = (split == 0) & (gold >= 0), (split == 1) & (gold >= 0)
    res = {"validation_resa2_minus_uniform": {}, "nyt_minus_released": {}}
    lines = ["# Bootstrap intervals (95%%, %d draws of the mentions, seeds averaged first)" % args.B, "",
             "## Validation (%d mentions): ReSa2 minus uniform, in points" % val.sum(), "", "| Measure | Difference | 95% interval |", "|---|---|---|"]
    for name, k in METRICS:
        m, lo, hi = boot(hits(ranks, seeds("resa2"), val, k), hits(ranks, seeds("uniform"), val, k), rng, args.B)
        res["validation_resa2_minus_uniform"][name] = [m, lo, hi]
        lines.append("| %s | %+.2f | %+.2f to %+.2f |" % (name, m, lo, hi))
    lines += ["", "## New York Times (%d mentions): each trained sampler minus the released model, in points" % nyt.sum(), "",
              "| Sampler | " + " | ".join(n for n, _ in METRICS) + " |", "|---|---|---|---|"]
    for s in ["in_batch", "uniform", "topscore", "resa2"]:
        cells = []
        res["nyt_minus_released"][s] = {}
        for name, k in METRICS:
            m, lo, hi = boot(hits(ranks, seeds(s), nyt, k), hits(ranks, ["released"], nyt, k), rng, args.B)
            res["nyt_minus_released"][s][name] = [m, lo, hi]
            cells.append("%+.1f (%+.1f to %+.1f)" % (m, lo, hi))
        lines.append("| %s | %s |" % (s, " | ".join(cells)))
    st = stage1_table()
    res["stage1"] = st
    lines += ["", "## What ReSa2 stage 1 keeps (200 validation pools, 20 draws each)", "", "| Measure | Share |", "|---|---|",
              "| Kept pages that are among the 500 closest in score to the correct page | %.0f%% |" % (100 * st["kept_pages_among_500_closest_in_score"]),
              "| Kept pages shared by two draws | %.0f%% |" % (100 * st["kept_pages_shared_by_two_draws"]),
              "| Pages after stage 2 that are among the 75 most similar pages of the whole pool | %.0f%% |" % (100 * st["stage2_pages_among_75_most_similar_of_pool"]),
              "| Pages after stage 2 shared by two draws | %.0f%% |" % (100 * st["stage2_pages_shared_by_two_draws"])]
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "bootstrap.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(args.out, "bootstrap.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
