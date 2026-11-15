"""Results of plan steps 1 and 5: weight mixing and the bandit run (added for resa2el). CPU only.

Run after scripts/plan_results.py RANKS_DIR (it adds the ranks to ranks_plan_compact.npz):
  python scripts/plan_results_mix.py
Writes docs/results/plan/steps1_5.md and steps1_5.json.
"""
import collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_results import COMPACT, OLD, PLAN, KS, hit, boot  # noqa: E402


def main():
    d, o = np.load(COMPACT), np.load(OLD)
    ranks = {k[5:]: d[k].astype(int) for k in d.files if k.startswith("rank_")}
    ranks.update({k[5:]: o[k].astype(int) for k in o.files if k.startswith("rank_")})
    split = np.where(d["gold"] >= 0, d["split"], -1)
    s3 = lambda n: [n + "_s%d" % i for i in range(3)]  # noqa: E731
    groups = collections.OrderedDict([
        ("released (0%)", ["released"]), ("mix 25%", s3("mix25")), ("mix 50% (main)", s3("mix50")), ("mix 75%", s3("mix75")),
        ("resa2 (100%)", s3("resa2")), ("average of the 3 resa2 models", ["soup_s012"]), ("all pool pages", ["allpool_s0"]), ("bandit", ["bandit_s0"])])
    rng = np.random.default_rng(0)
    res = {"table": {}, "diff": {}}
    L = ["# Plan steps 1 and 5: weight mixing and the bandit", "",
         "Mix: weights = (1 - a) x released + a x ReSa2 model, one mix for each of the 3 ReSa2 seeds; cells are the mean of the 3 mixes.",
         "The 50% mix was named as the main result before any of these numbers was seen.",
         "All numbers in percent. Validation has 16,692 mentions, NYT has 769. Intervals are 95% bootstrap intervals over mentions (seeds averaged first).", "",
         "| Model | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |", "|---|---|---|---|---|---|---|"]
    for g, names in groups.items():
        row = [round(float(hit(ranks, names, split == s, k).mean() * 100), 1) for s in (0, 1) for _, k in KS]
        res["table"][g] = row
        L.append("| " + g + " | " + " | ".join("%.1f" % x for x in row) + " |")
    pairs = [("mix 50% (main)", "released (0%)"), ("mix 50% (main)", "resa2 (100%)"), ("mix 75%", "released (0%)"), ("mix 75%", "resa2 (100%)"),
             ("mix 25%", "released (0%)"), ("average of the 3 resa2 models", "resa2 (100%)"), ("average of the 3 resa2 models", "released (0%)"),
             ("bandit", "resa2 (100%)")]
    L += ["", "## Differences", "", "| A minus B | split | top-1 | R@10 | R@100 |", "|---|---|---|---|---|"]
    for x, y in pairs:
        for s, sn in ((0, "valid"), (1, "NYT")):
            cells = []
            for kn, k in KS:
                b = boot(hit(ranks, groups[x], split == s, k), hit(ranks, groups[y], split == s, k), rng)
                res["diff"]["%s - %s | %s | %s" % (x, y, sn, kn)] = b
                cells.append("%+.2f (%+.2f to %+.2f)" % tuple(b))
            L.append("| %s minus %s | %s | %s |" % (x, y, sn, " | ".join(cells)))
    f = os.path.join(PLAN, "bandit_s0_rewards.jsonl")
    if os.path.exists(f):
        r = [json.loads(l) for l in open(f)][1:]
        arms = ["top25", "25to75", "75to300", "rest"]
        L += ["", "## Bandit (EXP3 over the 4 bands, 150 rounds of 10 updates)", "",
              "| Band | picks, rounds 1 to 50 | 51 to 100 | 101 to 150 | all | mean scaled reward | probability at the end |", "|---|---|---|---|---|---|---|"]
        res["bandit"] = {}
        for i, a in enumerate(arms):
            c = [sum(x["arm"] == a for x in r[lo:hi]) for lo, hi in ((0, 50), (50, 100), (100, 150), (0, 150))]
            ms = float(np.mean([x["scaled"] for x in r if x["arm"] == a]))
            res["bandit"][a] = c + [round(ms, 3), r[-1]["probs"][i]]
            L.append("| %s | %d | %d | %d | %d | %.2f | %.2f |" % (a, c[0], c[1], c[2], c[3], ms, r[-1]["probs"][i]))
        L += ["", "Highest probability of any band in any round: %.2f. Held-out loss at the end: %.3f." % (max(max(x["probs"]) for x in r), r[-1]["heldout_loss"])]
    open(os.path.join(PLAN, "steps1_5.md"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(PLAN, "steps1_5.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
