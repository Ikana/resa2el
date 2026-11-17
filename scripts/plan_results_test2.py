"""Second test round (docs/results/plan/test_list2.md): table and the fixed comparisons (added for resa2el).

Run: python scripts/plan_results_test2.py ROUND1_TEST_RANKS ROUND2_TEST_RANKS ROUND2_VALID_RANKS
Writes docs/results/plan/test2.md and .json. Paired bootstrap over mentions, mean over the seeds first.
"""
import json
import sys

import numpy as np

B = 10000


def hits(path, key="rank", gkey="gold"):
    z = np.load(path, allow_pickle=True)
    r = z[key][z[gkey] >= 0]
    return np.stack([r == 0, (r >= 0) & (r < 10), r >= 0]).astype(float)   # 3 x n


def row(d, names):
    h = np.stack([hits("{}/{}.npz".format(d, n)) for n in names])            # seeds x 3 x n
    per = 100 * h.mean(2)
    return h.mean(0), per.mean(0), (per.std(0, ddof=1) if len(names) > 1 else None)


def cell(mean, sd):
    return " | ".join("{:.1f}".format(m) if sd is None else "{:.1f} ± {:.1f}".format(m, s) for m, s in zip(mean, sd if sd is not None else mean))


def boot(a, b, rng):
    d = 100 * (a - b)                                                         # 3 x n
    n = d.shape[1]
    m = np.concatenate([d[:, rng.integers(0, n, size=(500, n))].mean(2) for _ in range(B // 500)], axis=1)   # 3 x B
    return d.mean(1), np.percentile(m, [2.5, 97.5], axis=1).T, np.percentile(m, [0.5, 99.5], axis=1).T


def main():
    d1, d2, dv = sys.argv[1:4]
    rng = np.random.default_rng(0)
    S3 = ["_s0", "_s1", "_s2"]
    rows = {"Released EVELINK (round 1)": row(d1, ["released"]),
            "ReSa2 (round 1)": row(d1, ["resa2" + s for s in S3]),
            "All pool pages (round 1)": row(d1, ["allpool_s0"]),
            "ReSa2 mixed 50% (round 1)": row(d1, ["mix50" + s for s in S3]),
            "Live hard negatives (round 2)": row(d2, ["livetop" + s for s in S3]),
            "Live hard negatives mixed 50% (round 2)": row(d2, ["livetopmix50_s1", "livetopmix50_s2"])}
    n = rows["Released EVELINK (round 1)"][0].shape[1]
    L = ["# Second test round: results", "",
         "Wikipedia test split, {} mentions. The list was fixed in `test_list2.md` before the evaluations.".format(n),
         "Round 1 rows are copied from `step7_test.md` and were not run again.", "",
         "| Row | Top-1 | R@10 | R@100 |", "|---|---|---|---|"]
    out = {"n": n, "rows": {}, "diffs": {}}
    for k, (_, mean, sd) in rows.items():
        L.append("| {} | {} |".format(k, cell(mean, sd)))
        out["rows"][k] = {"mean": mean.tolist(), "sd": None if sd is None else sd.tolist()}
    L += ["", "## Comparisons fixed in the list", "",
          "Difference in points, paired bootstrap over mentions ({} resamples), mean over the seeds first.".format(B), "",
          "| Comparison | Metric | Difference | 95% interval | 99% interval |", "|---|---|---|---|---|"]
    lt, mx = "Live hard negatives (round 2)", "Live hard negatives mixed 50% (round 2)"
    for a, b in [(lt, "ReSa2 (round 1)"), (mx, "ReSa2 mixed 50% (round 1)"), (lt, "Released EVELINK (round 1)"), (mx, "Released EVELINK (round 1)")]:
        m, c95, c99 = boot(rows[a][0], rows[b][0], rng)
        name = "{} minus {}".format(a.split(" (")[0], b.split(" (")[0])
        out["diffs"][name] = {"diff": m.tolist(), "ci95": c95.tolist(), "ci99": c99.tolist()}
        for i, met in enumerate(["Top-1", "R@10"]):
            L.append("| {} | {} | {:+.2f} | {:+.2f} to {:+.2f} | {:+.2f} to {:+.2f} |".format(name, met, m[i], c95[i][0], c95[i][1], c99[i][0], c99[i][1]))
    L += ["", "## The models trained again: validation and NYT", "",
          "| Seed | Valid top-1 | Valid R@10 | Valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |", "|---|---|---|---|---|---|---|"]
    vs = []
    for s in S3:
        z = np.load("{}/livetop{}.npz".format(dv, s), allow_pickle=True)
        v = []
        for p in (0, 1):   # split code: 0 validation, 1 NYT
            r = z["rank"][(z["split"] == p) & (z["gold"] >= 0)]
            v += [100 * (r == 0).mean(), 100 * ((r >= 0) & (r < 10)).mean(), 100 * (r >= 0).mean()]
        vs.append(v)
        L.append("| {} | {} |".format(s[2:], " | ".join("{:.1f}".format(x) for x in v)))
    a = np.array(vs)
    L.append("| Mean | {} |".format(" | ".join("{:.1f} ± {:.1f}".format(m, sd) for m, sd in zip(a.mean(0), a.std(0, ddof=1)))))
    out["retrained_valid_nyt"] = vs
    open("docs/results/plan/test2.md", "w").write("\n".join(L) + "\n")
    json.dump(out, open("docs/results/plan/test2.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
