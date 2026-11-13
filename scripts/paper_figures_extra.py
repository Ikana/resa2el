"""Four more figures for the paper (added for resa2el). CPU only.

Run: python scripts/paper_figures_extra.py --out docs/paper/figures
  Reads docs/results/gaps/gaps.json (per-run recall and mistakes) and
  docs/results/analysis/pools_sample200.npz (candidate pools of 200 random
  validation mentions, copied from out/pools/valid.npz).

Writes draws_one_mention.pdf, before_after.pdf, mistakes.pdf, by_mention_type.pdf.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from resa2el import samplers as S  # noqa: E402

COLOR = {"uniform": "#0072B2", "topscore": "#E69F00", "resa2": "#009E73", "in_batch": "#999999", "released": "#444444"}
LABEL = {"uniform": "Uniform", "topscore": "Top-score", "resa2": "ReSa2", "in_batch": "In-batch", "released": "Released"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.5, "legend.frameon": False})
SAMPLERS = ["in_batch", "uniform", "topscore", "resa2"]


def mean(runs, s, f):
    return float(np.mean([f(runs[k]) for k in runs if k.startswith(s + "_s")]))


def draws_one_mention(d, out, n_draws=20, mention=None):
    """Twenty draws of seven negatives for one validation mention, by similarity rank."""
    rng = np.random.default_rng(0)
    if mention is None:  # the mention whose ReSa2 coverage over 200 draws is closest to the average of 84
        best = None
        for i in range(len(d["gold"])):
            sc, sims, sp = pool(d, i)
            cov = len({int(x) for _ in range(200) for x in S.resa2(sc, sp, sims, 7, rng)})
            if best is None or abs(cov - 84) < best[0]:
                best = (abs(cov - 84), i)
        mention = best[1]
    sc, sims, sp = pool(d, mention)
    rank = np.empty(len(sims), int)
    rank[np.argsort(-sims, kind="stable")] = np.arange(1, len(sims) + 1)
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(3, 1, figsize=(3.9, 2.07), sharex=True)
    for ax, name in zip(axes, ["topscore", "uniform", "resa2"]):
        for r in range(n_draws):
            sel = S.SAMPLERS[name](sc, sp, sims, 7, rng)
            ax.scatter(rank[sel], np.full(7, r), s=5, color=COLOR[name], linewidths=0)
        ax.set_xscale("log"); ax.set_xlim(0.8, 1100); ax.set_ylim(n_draws - 0.3, -0.7)
        ax.set_yticks([]); ax.grid(axis="y", visible=False)
        ax.set_ylabel(LABEL[name], rotation=0, ha="right", va="center")
    axes[-1].set_xticks([1, 10, 100, 1000]); axes[-1].set_xticklabels(["1", "10", "100", "1,000"])
    axes[-1].set_xlabel("similarity rank (1 = most like the correct page)")
    fig.tight_layout(h_pad=0.4); fig.savefig(os.path.join(out, "draws_one_mention.pdf")); plt.close(fig)
    return int(d["idx"][mention])


def pool(d, i):
    keep = d["ids"][i] != d["gold"][i]
    return d["scores"][i][keep].astype(np.float64), d["sims"][i][keep].astype(np.float64), float(d["gold_score"][i])


def before_after(runs, out):
    """Recall at 10 of the released retriever and of each trained one, on two data sets."""
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.0), sharey=True)
    panels = [("Wikipedia validation", lambda r: r["Validation, all"][0]), ("New York Times", lambda r: r["NYT"]["r10"])]
    for ax, (title, f) in zip(axes, panels):
        base = f(runs["released"])
        ends = sorted(((mean(runs, s, f), s) for s in SAMPLERS), reverse=True)
        last = 200.0
        for v, s in ends:
            ax.plot([0, 1], [base, v], color=COLOR[s], linewidth=1.8, marker="o", markersize=4)
            y = min(v, last - 8.5); last = y
            ax.text(1.13, y, "%s %.1f" % (LABEL[s], v), va="center", fontsize=6.5, color=COLOR[s], fontweight="bold")
        ax.plot([0], [base], marker="o", markersize=5, color=COLOR["released"])
        ax.text(-0.13, base, "%.1f" % base, ha="right", va="center", fontsize=6.5)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlim(-0.6, 2.75); ax.set_xticks([0, 1]); ax.set_xticklabels(["released", "trained"])
        for lab, ha in zip(ax.get_xticklabels(), ["right", "left"]):
            lab.set_ha(ha)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("recall at 10 (%)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "before_after.pdf")); plt.close(fig)


def mistakes(runs, out):
    """Mentions whose first result is wrong, split by kind of mistake."""
    kinds = [("time", "Time", "#D55E00"), ("hierarchy", "Hierarchy", "#56B4E9"), ("other", "Other", "#cfcfcf")]
    n = runs["released"]["Validation, all"][1]
    rows = [("Released", {k: runs["released"]["errors"][k] / n * 100 for k, _, _ in kinds})]
    rows += [(LABEL[s], {k: mean(runs, s, lambda r, k=k: r["errors"][k] / n * 100) for k, _, _ in kinds}) for s in SAMPLERS]
    fig, ax = plt.subplots(figsize=(3.9, 1.65))
    for y, (name, v) in enumerate(rows):
        left = 0.0
        for k, lab, c in kinds:
            ax.barh(y, v[k], left=left, color=c, edgecolor="white", linewidth=1, height=0.62, label=lab if y == 0 else None)
            left += v[k]
        ax.text(left + 0.8, y, "%.1f  (time %.1f)" % (left, v["time"]), va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("wrong first result (% of validation mentions)")
    ax.set_xlim(0, 92); ax.grid(axis="y", visible=False)
    fig.legend(ncol=3, loc="upper center", fontsize=6.5, handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(out, "mistakes.pdf")); plt.close(fig)


def by_mention_type(runs, out):
    """Recall at 10 for each kind of mention, one dot per retriever."""
    classes = [("Verb, seen", "Verb, seen"), ("Verb, unseen form", "Verb, unseen form"), ("Verb, unseen", "Verb, unseen"),
               ("Nominal, hard", "Noun, hard"), ("Nominal, easy", "Noun, easy")]
    fig, ax = plt.subplots(figsize=(3.9, 1.95))
    for y, (key, lab) in enumerate(classes):
        vals = {"released": runs["released"][key][0]}
        vals.update({s: mean(runs, s, lambda r, key=key: r[key][0]) for s in SAMPLERS})
        ax.plot([min(vals.values()), max(vals.values())], [y, y], color="#d0d0d0", linewidth=1.5, zorder=1)
        for s, v in vals.items():
            ax.scatter([v], [y], s=18, color=COLOR[s], edgecolor="white", linewidth=0.8, zorder=2, label=LABEL[s] if y == 0 else None)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([c[1] for c in classes]); ax.invert_yaxis()
    ax.set_xlim(50, 101); ax.set_xlabel("recall at 10 (%)"); ax.grid(axis="y", visible=False)
    fig.legend(ncol=5, loc="upper center", fontsize=6.5, handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(out, "by_mention_type.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "paper", "figures"))
    ap.add_argument("--mention", type=int, default=None, help="row of the sample file to draw (default: chosen by rule)")
    args = ap.parse_args()
    runs = json.load(open(os.path.join(ROOT, "docs", "results", "gaps", "gaps.json")))
    d = np.load(os.path.join(ROOT, "docs", "results", "analysis", "pools_sample200.npz"))
    print("mention drawn: validation row", draws_one_mention(d, args.out, mention=args.mention))
    before_after(runs, args.out); mistakes(runs, args.out); by_mention_type(runs, args.out)


if __name__ == "__main__":
    main()
