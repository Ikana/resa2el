"""Step 7 of the results plan: table of the test-set evaluations (added for resa2el).

Run: python scripts/plan_results_test.py RANKS_DIR
Reads RANKS_DIR/NAME.npz (from eval_ranks.py --test), writes docs/results/plan/step7_test.md and .json.
"""
import json
import sys

import numpy as np

ROWS = [("Released EVELINK", ["released"]),
        ("Uniform from the pool, 7 negatives", ["uniform_s0", "uniform_s1", "uniform_s2"]),
        ("ReSa2, 7 negatives", ["resa2_s0", "resa2_s1", "resa2_s2"]),
        ("Stage 2 only, 7 negatives", ["stage2only_s0", "stage2only_s1", "stage2only_s2"]),
        ("All pool pages (about 999 negatives)", ["allpool_s0"]),
        ("ReSa2 mixed 50% with the released model", ["mix50_s0", "mix50_s1", "mix50_s2"])]


def metrics(path):
    z = np.load(path, allow_pickle=True)
    r = z["rank"][z["gold"] >= 0]
    return len(r), [100 * float((r == 0).mean()), 100 * float(((r >= 0) & (r < 10)).mean()), 100 * float((r >= 0).mean())]


def main():
    d = sys.argv[1]
    out, lines = {}, ["# Step 7: test set results", "",
                      "Wikipedia test split. Each model was evaluated one time (list fixed in `test_list.md`).",
                      "Mean over the seeds, with the standard deviation over the seeds (n - 1) where there are 3 seeds.", "",
                      "| Row | Seeds | Top-1 | R@10 | R@100 |", "|---|---|---|---|---|"]
    for label, names in ROWS:
        ms = []
        for nm in names:
            n, m = metrics("{}/{}.npz".format(d, nm))
            ms.append(m)
        a = np.array(ms)
        mean, sd = a.mean(0), (a.std(0, ddof=1) if len(names) > 1 else None)
        out[label] = {"n": n, "seeds": len(names), "mean": mean.tolist(), "sd": None if sd is None else sd.tolist(), "per_seed": ms}
        cell = ["{:.1f}".format(x) if sd is None else "{:.1f} ± {:.1f}".format(x, s) for x, s in zip(mean, sd if sd is not None else mean)]
        lines.append("| {} | {} | {} |".format(label, len(names), " | ".join(cell)))
    lines += ["", "Mentions: {}.".format(n)]
    open("docs/results/plan/step7_test.md", "w").write("\n".join(lines) + "\n")
    json.dump(out, open("docs/results/plan/step7_test.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
