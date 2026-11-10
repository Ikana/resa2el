"""Per-class recall, error types, and NYT recall from the per-mention ranks (added for resa2el).

Run: python scripts/analyze_gaps.py RANKS_DIR --out docs/results/gaps
  RANKS_DIR holds released.npz and <sampler>_s<seed>.npz from scripts/eval_ranks.py.

Writes gaps.md (three tables) and gaps.json.

Error types (operational definitions, on validation mentions whose top-1 page
is wrong and whose correct page is in the store):
  time       the predicted title equals the correct title after all 4-digit
             years are removed, and the years differ ("1998 AFL Rising Star"
             for "1996 AFL Rising Star")
  hierarchy  the words of the shorter title are a subset of the words of the
             longer one ("National Football League Draft" vs "2000 NFL Draft"
             does not match this rule; "2000 NFL Draft" vs "NFL Draft" does)
  other      everything else
"""
import argparse
import glob
import json
import os
import re
import statistics as st

import numpy as np

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def norm(title):
    t = title.strip()
    if t.startswith("'") and t.endswith("'"):
        t = t[1:-1]
    return t.replace("_", " ").lower()


def error_type(pred, gold):
    p, g = norm(pred), norm(gold)
    py, gy = YEAR.findall(p), YEAR.findall(g)
    if YEAR.sub("", p).split() == YEAR.sub("", g).split() and py != gy and (py or gy):
        return "time"
    pw, gw = set(YEAR.sub("", p).split()), set(YEAR.sub("", g).split())
    if pw and gw and (pw < gw or gw < pw):
        return "hierarchy"
    return "other"


CLASSES = [("verb", "seen_event_seen_form", "Verb, seen"), ("verb", "seen_event_unseen_form", "Verb, unseen form"),
           ("verb", "unseen_event", "Verb, unseen"), ("other", "hard", "Nominal, hard"), ("other", "easy", "Nominal, easy")]


def per_run(path):
    d = np.load(path, allow_pickle=True)
    split, gold, rank = d["split"], d["gold"], d["rank"].astype(int)
    mtype, status = d["mtype"], d["status"]
    ok = gold >= 0
    out = {}
    v = (split == 0) & ok
    for t, s_, label in CLASSES:
        m = v & (mtype == t) & (status == s_)
        out[label] = (float(((rank[m] >= 0) & (rank[m] < 10)).mean()) * 100, int(m.sum()))
    out["Validation, all"] = (float(((rank[v] >= 0) & (rank[v] < 10)).mean()) * 100, int(v.sum()))
    n = (split == 1) & ok
    out["NYT"] = {"top1": float((rank[n] == 0).mean()) * 100, "r10": float(((rank[n] >= 0) & (rank[n] < 10)).mean()) * 100,
                  "r100": float((rank[n] >= 0).mean()) * 100, "n": int(n.sum())}
    wrong = v & (rank != 0)
    types = {"time": 0, "hierarchy": 0, "other": 0}
    for i in np.flatnonzero(wrong):
        types[error_type(str(d["titles10"][i][0]), str(d["gold_title"][i]))] += 1
    out["errors"] = {"n_wrong": int(wrong.sum()), **types}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks_dir")
    ap.add_argument("--out", default="docs/results/gaps")
    args = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(args.ranks_dir, "*.npz"))):
        runs[os.path.basename(f)[:-4]] = per_run(f)
    samplers = ["in_batch", "uniform", "topscore", "resa2"]
    def seeds(s):
        return [runs[k] for k in runs if k.startswith(s + "_s")]
    def mean_sd(vals):
        return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)
    lines = ["# Per-class recall, error types, and NYT (lr 1e-5 runs, mean ± sd over seeds)", "",
             "## Recall@10 by mention class (validation)", "",
             "| Model | " + " | ".join(label for _, _, label in CLASSES) + " | All |", "|---|" + "---|" * (len(CLASSES) + 1)]
    rel = runs.get("released")
    if rel:
        lines.append("| Released | " + " | ".join("%.1f" % rel[label][0] for _, _, label in CLASSES) + " | %.1f |" % rel["Validation, all"][0])
    for s_ in samplers:
        rs = seeds(s_)
        if not rs: continue
        cells = []
        for _, _, label in CLASSES:
            m, sd = mean_sd([r[label][0] for r in rs]); cells.append("%.1f ± %.1f" % (m, sd))
        m, sd = mean_sd([r["Validation, all"][0] for r in rs])
        lines.append("| %s | %s | %.1f ± %.1f |" % (s_, " | ".join(cells), m, sd))
    if rel:
        lines += ["", "Class sizes: " + ", ".join("%s %d" % (label, rel[label][1]) for _, _, label in CLASSES) + "."]
    lines += ["", "## Top-1 errors by type (validation; share of all mentions)", "", "| Model | Wrong top-1 | Time | Hierarchy | Other |", "|---|---|---|---|---|"]
    def err_row(name, rs):
        n = rs[0]["Validation, all"][1]
        vals = {k: mean_sd([r["errors"][k] / n * 100 for r in rs]) for k in ("n_wrong", "time", "hierarchy", "other")}
        return "| %s | %s |" % (name, " | ".join("%.1f ± %.1f" % vals[k] if len(rs) > 1 else "%.1f" % vals[k][0] for k in ("n_wrong", "time", "hierarchy", "other")))
    if rel: lines.append(err_row("Released", [rel]))
    for s_ in samplers:
        rs = seeds(s_)
        if rs: lines.append(err_row(s_, rs))
    lines += ["", "## New York Times (mentions with a page, no Nil)", "", "| Model | Top-1 | Recall@10 | Recall@100 |", "|---|---|---|---|"]
    def nyt_row(name, rs):
        cells = []
        for k in ("top1", "r10", "r100"):
            m, sd = mean_sd([r["NYT"][k] for r in rs]); cells.append("%.1f ± %.1f" % (m, sd) if len(rs) > 1 else "%.1f" % m)
        return "| %s | %s |" % (name, " | ".join(cells))
    if rel: lines.append(nyt_row("Released", [rel]) + "  (n = %d)" % rel["NYT"]["n"])
    for s_ in samplers:
        rs = seeds(s_)
        if rs: lines.append(nyt_row(s_, rs))
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "gaps.md"), "w").write("\n".join(lines) + "\n")
    json.dump(runs, open(os.path.join(args.out, "gaps.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
