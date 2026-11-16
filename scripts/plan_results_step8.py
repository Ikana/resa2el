"""Results of the four new samplers of 2026-11-16 (added for resa2el). CPU only.

Run: python scripts/plan_results_step8.py [RANKS_DIR]
With RANKS_DIR, the ranks there are first added to docs/results/plan/ranks_plan_compact.npz.
Writes docs/results/plan/step8.md and step8.json.
"""
import glob, json, os, sys
import numpy as np
from plan_results import COMPACT, OLD, PLAN, KS, hit, boot

GROUPS = [("released", ["released"]),
          ("resa2 (3 seeds)", ["resa2_s0", "resa2_s1", "resa2_s2"]),
          ("all pool pages, 999 negatives", ["allpool_s0"]),
          ("score-proportional with correction", ["scoreprop_s0"]),
          ("mixed hard and random", ["mixed_s0"]),
          ("live top below the correct page", ["livetopbelow_s0"]),
          ("live top, seed 0", ["livetop_s0"]),
          ("live top (3 seeds)", ["livetop_s0", "livetop_s1", "livetop_s2"]),
          ("resa2 mixed 50% (3 seeds)", ["mix50_s0", "mix50_s1", "mix50_s2"]),
          ("live top mixed 50% (seeds 1, 2)", ["livetopmix50_s1", "livetopmix50_s2"])]
DIFFS = [("live top (3 seeds)", "resa2 (3 seeds)"), ("live top (3 seeds)", "all pool pages, 999 negatives"),
         ("score-proportional with correction", "resa2 (3 seeds)"), ("score-proportional with correction", "live top, seed 0"),
         ("live top below the correct page", "live top, seed 0"), ("mixed hard and random", "resa2 (3 seeds)"),
         ("live top (3 seeds)", "released"),
         ("live top mixed 50% (seeds 1, 2)", "released"), ("live top mixed 50% (seeds 1, 2)", "resa2 mixed 50% (3 seeds)")]


def main():
    if len(sys.argv) > 1:
        out = dict(np.load(COMPACT))
        for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
            d = np.load(f, allow_pickle=True)
            assert (d["gold"] == out["gold"]).all()
            out["rank_" + os.path.basename(f)[:-4]] = d["rank"].astype(np.int16)
        np.savez_compressed(COMPACT, **out)
    d, o = np.load(COMPACT), np.load(OLD)
    ranks = {k[5:]: d[k].astype(int) for k in d.files if k.startswith("rank_")}
    ranks.update({k[5:]: o[k].astype(int) for k in o.files if k.startswith("rank_")})
    split = np.where(d["gold"] >= 0, d["split"], -1)
    G = dict(GROUPS)
    rng = np.random.default_rng(0)
    res = {"table": {}, "sd": {}, "diff": {}}
    L = ["# Four new samplers (2026-11-16)", "",
         "7 negatives for each mention, 1,500 updates, same setup as the other runs. All numbers in percent.",
         "Validation has 16,692 mentions, NYT has 769. A cell with ± is the mean and the standard deviation over the seeds.",
         "Intervals are 95% bootstrap intervals over mentions, with the seeds averaged first.", "",
         "| Run | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |", "|---|---|---|---|---|---|---|"]
    for g, names in GROUPS:
        per = np.array([[hit(ranks, [n], split == s, k).mean() * 100 for s in (0, 1) for _, k in KS] for n in names])
        res["table"][g] = [round(float(x), 2) for x in per.mean(0)]
        if len(names) > 1:
            res["sd"][g] = [round(float(x), 2) for x in per.std(0, ddof=1)]
            cells = ["%.1f ± %.1f" % (m, s) for m, s in zip(per.mean(0), per.std(0, ddof=1))]
        else:
            cells = ["%.1f" % m for m in per.mean(0)]
        L.append("| " + g + " | " + " | ".join(cells) + " |")
    L += ["", "## Differences (first minus second, percentage points, with interval)", "",
          "| Difference | valid top-1 | valid R@10 | NYT R@10 | NYT R@100 |", "|---|---|---|---|---|"]
    for a, b in DIFFS:
        cells = []
        for s, k in ((0, 1), (0, 10), (1, 10), (1, 100)):
            v = boot(hit(ranks, G[a], split == s, k), hit(ranks, G[b], split == s, k), rng)
            res["diff"].setdefault(a + " - " + b, {})["%s@%d" % ("valid" if s == 0 else "nyt", k)] = v
            cells.append("%+.2f (%+.2f to %+.2f)" % tuple(v))
        L.append("| " + a + " − " + b + " | " + " | ".join(cells) + " |")
    open(os.path.join(PLAN, "step8.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(PLAN, "step8.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
