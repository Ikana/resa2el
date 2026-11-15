"""Results of the plan runs (steps 2 to 6): table, bootstrap differences, reward log summary (added for resa2el). CPU only.

Run: python scripts/plan_results.py [RANKS_DIR] [--rewards DIR]
  RANKS_DIR  folder with <name>_s<seed>.npz from scripts/eval_ranks.py. With it, the script
             first adds these ranks to docs/results/plan/ranks_plan_compact.npz.
  --rewards  folder with <name>_s<seed>.jsonl reward logs. With it, the script copies a
             summary to docs/results/plan/rewards.json.
Writes docs/results/plan/steps4_6.md and steps4_6.json.
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PLAN = os.path.join(ROOT, "docs", "results", "plan")
COMPACT = os.path.join(PLAN, "ranks_plan_compact.npz")
OLD = os.path.join(ROOT, "docs", "results", "gaps", "ranks_compact.npz")
KS = [("top-1", 1), ("R@10", 10), ("R@100", 100)]


def hit(ranks, names, sel, k):
    return np.mean([((ranks[n][sel] >= 0) & (ranks[n][sel] < k)) for n in names], axis=0)


def boot(a, b, rng, B=2000):
    diff = (a - b) * 100
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    m = diff[idx].mean(axis=1)
    return [round(float(diff.mean()), 2), round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks_dir", nargs="?")
    ap.add_argument("--rewards")
    a = ap.parse_args()
    if a.ranks_dir:
        out = dict(np.load(COMPACT)) if os.path.exists(COMPACT) else {}
        for f in sorted(glob.glob(os.path.join(a.ranks_dir, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            out["rank_" + os.path.basename(f)[:-4]] = d["rank"].astype(np.int16)
            out["split"], out["gold"] = d["split"], d["gold"]
        np.savez_compressed(COMPACT, **out)
    d, o = np.load(COMPACT), np.load(OLD)
    ranks = {k[5:]: d[k].astype(int) for k in d.files if k.startswith("rank_")}
    ranks.update({k[5:]: o[k].astype(int) for k in o.files if k.startswith("rank_")})
    split = np.where(d["gold"] >= 0, d["split"], -1)  # NYT rows without a correct page in the index are left out
    groups = {"released": ["released"], "resa2 (3 seeds)": ["resa2_s0", "resa2_s1", "resa2_s2"],
              "stage2only (3 seeds)": ["stage2only_s0", "stage2only_s1", "stage2only_s2"],
              "band top 25": ["band_top25_s0"], "band 25 to 75": ["band_25to75_s0"],
              "band 75 to 300": ["band_75to300_s0"], "band rest": ["band_rest_s0"]}
    rng = np.random.default_rng(0)
    res = {"table": {}, "diff": {}, "seeds": {}}
    L = ["# Plan steps 4 and 6: similarity bands and stage 2 only", "",
         "All numbers in percent. Validation has 16,692 mentions, NYT has 769. Intervals are 95% bootstrap intervals over mentions.", "",
         "| Run | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |", "|---|---|---|---|---|---|---|"]
    for g, names in groups.items():
        row = [round(float(hit(ranks, names, split == s, k).mean() * 100), 1) for s in (0, 1) for _, k in KS]
        res["table"][g] = row
        L.append("| " + g + " | " + " | ".join("%.1f" % x for x in row) + " |")
    for n in groups["stage2only (3 seeds)"] + groups["resa2 (3 seeds)"]:
        res["seeds"][n] = [round(float(hit(ranks, [n], split == s, k).mean() * 100), 1) for s in (0, 1) for _, k in KS]
    pairs = [("stage2only (3 seeds)", "resa2 (3 seeds)"), ("band top 25", "band 25 to 75"), ("band 25 to 75", "band 75 to 300"),
             ("band 75 to 300", "band rest"), ("band top 25", "band rest"), ("band top 25", "resa2 (3 seeds)"),
             ("stage2only (3 seeds)", "released"), ("band rest", "released")]
    L += ["", "## Differences", "", "| A minus B | split | top-1 | R@10 | R@100 |", "|---|---|---|---|---|"]
    for x, y in pairs:
        for s, sn in ((0, "valid"), (1, "NYT")):
            cells = []
            for kn, k in KS:
                b = boot(hit(ranks, groups[x], split == s, k), hit(ranks, groups[y], split == s, k), rng)
                res["diff"]["%s - %s | %s | %s" % (x, y, sn, kn)] = b
                cells.append("%+.2f (%+.2f to %+.2f)" % tuple(b))
            L.append("| %s minus %s | %s | %s |" % (x, y, sn, " | ".join(cells)))
    L += ["", "## Seeds", "", "| Run | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |", "|---|---|---|---|---|---|---|"]
    for n, row in res["seeds"].items():
        L.append("| " + n + " | " + " | ".join("%.1f" % x for x in row) + " |")
    rw_file = os.path.join(PLAN, "rewards.json")
    if a.rewards:
        rw = {}
        for f in sorted(glob.glob(os.path.join(a.rewards, "*.jsonl"))):
            rows = [json.loads(l) for l in open(f)]
            rw[os.path.basename(f)[:-6]] = {"update": [r["update"] for r in rows], "heldout_loss": [round(r["heldout_loss"], 4) for r in rows]}
        json.dump(rw, open(rw_file, "w"))
    if os.path.exists(rw_file):
        rw = json.load(open(rw_file))
        L += ["", "## Reward log (step 4)", "",
              "Held-out loss: cross entropy against the full pool on 128 held-out training mentions, measured each 10 updates. The reward of an interval is the fall of this loss.", "",
              "| Run | loss at 0 | at 100 | at 300 | at 750 | at 1,500 | lowest (update) | mean reward 0 to 300 | mean reward 300 to 1,500 | SD of reward 300 to 1,500 |", "|---|---|---|---|---|---|---|---|---|---|"]
        res["rewards"] = {}
        for n, r in rw.items():
            u, h = np.array(r["update"]), np.array(r["heldout_loss"])
            at = lambda x: float(h[np.argmin(np.abs(u - x))])
            rew = -np.diff(h)
            early, late = rew[u[1:] <= 300], rew[u[1:] > 300]
            row = [at(0), at(100), at(300), at(750), float(h[-1]), float(h.min()), int(u[h.argmin()]), float(early.mean()), float(late.mean()), float(late.std())]
            res["rewards"][n] = row
            L.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f (%d) | %+.4f | %+.4f | %.4f |" % tuple([n] + row))
    open(os.path.join(PLAN, "steps4_6.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(PLAN, "steps4_6.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
