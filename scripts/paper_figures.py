"""Figures for the paper (added for resa2el). CPU only.

Run: python scripts/paper_figures.py POOLS.npz LOGDIR --out docs/paper/figures
  POOLS.npz  validation pools from scripts/build_pools.py
  LOGDIR     folder with log_<sampler>_s<seed>.txt training logs (healthy and collapsed)

Writes weight_window.pdf, inclusion_by_rank.pdf, loss_curves.pdf.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from resa2el import samplers as S  # noqa: E402

# One color per sampler, fixed everywhere (Okabe-Ito, colorblind-safe).
COLOR = {"uniform": "#0072B2", "topscore": "#E69F00", "resa2": "#009E73", "in_batch": "#999999"}
LABEL = {"uniform": "Uniform", "topscore": "Top-score", "resa2": "ReSa2", "in_batch": "In-batch"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.5, "legend.frameon": False})


def weight_window(d, out):
    """Stage-1 weight against the score gap, over the pool's score-gap histogram."""
    gap = (d["scores"][:2000].astype(np.float64) - d["gold_score"][:2000, None]).ravel()
    gap = gap[np.isfinite(gap)]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.hist(gap, bins=np.arange(-60, 21, 1), density=True, color="#cfcfcf", edgecolor="none", label="pool pages, share per 1-point bin")
    x = np.linspace(-60, 20, 800)
    ax2 = ax.twinx() if False else ax  # one axis: weight is drawn on the density scale, normalized to the peak
    w = np.exp(-0.25 * x ** 2)
    ax.plot(x, w * ax.get_ylim()[1] * 0.80, color=COLOR["resa2"], linewidth=2, label="stage-1 weight $e^{-0.25 (s_i - s^+)^2}$, scaled")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.text(1.0, ax.get_ylim()[1] * 0.04, "correct\npage", fontsize=7, va="bottom", color="#555555")
    ax.set_xlabel("score minus the correct page's score, $s_i - s^+$")
    ax.set_ylabel("share of pool pages")
    ax.set_xlim(-60, 20)
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "weight_window.pdf"))
    plt.close(fig)


def inclusion_by_rank(d, out, mentions=300, repeats=100, k=7):
    ok = np.flatnonzero(d["gold"] >= 0)
    picks = np.random.default_rng(0).choice(ok, mentions, replace=False)
    bins = np.array([0, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 1000])
    prob = {n: np.zeros(len(bins) - 1) for n in ("uniform", "topscore", "resa2")}
    for i in picks:
        not_gold = d["ids"][i] != d["gold"][i]
        scores = d["scores"][i][not_gold].astype(np.float64)
        sims = d["sims"][i][not_gold].astype(np.float64)
        s_pos = float(d["gold_score"][i])
        rank = np.empty(len(scores), dtype=int)
        rank[np.argsort(-sims, kind="stable")] = np.arange(len(scores))
        b = np.digitize(rank, bins) - 1
        for name in prob:
            counts = np.zeros(len(scores))
            for r in range(repeats):
                counts[np.asarray(S.SAMPLERS[name](scores, s_pos, sims, k, np.random.default_rng(r)))] += 1
            incl = counts / repeats
            for j in range(len(bins) - 1):
                m = b == j
                if m.any():
                    prob[name][j] += incl[m].mean()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    centers = np.sqrt(bins[:-1].clip(1) * bins[1:])
    for name in ("uniform", "topscore", "resa2"):
        y = prob[name] / mentions
        ax.plot(centers, y, color=COLOR[name], linewidth=2, marker="o", markersize=3, label=LABEL[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("similarity rank (1 = most similar)")
    ax.set_ylabel("P(page is selected)")
    ax.axhline(k / 1000, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(300, k / 1000 * 1.3, "k/N = 0.007", fontsize=7, color="#777777")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_ylim(5e-5, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "inclusion_by_rank.pdf"))
    plt.close(fig)


def loss_curves(logdir, out):
    runs = {"resa2 seed 0 (healthy)": ("resa2_s0", COLOR["resa2"], "-"),
            "resa2 seed 1 (collapsed)": ("resa2_s1", COLOR["resa2"], "--"),
            "topscore seed 1 (collapsed)": ("topscore_s1", COLOR["topscore"], "--"),
            "uniform seed 1": ("uniform_s1", COLOR["uniform"], "-")}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for label, (name, color, ls) in runs.items():
        path = os.path.join(logdir, "log_{}.txt".format(name))
        if not os.path.exists(path):
            continue
        steps, losses = [], []
        for line in open(path):
            m = re.search(r"Step (\d+) - epoch 0 average loss: ([0-9.]+)", line)
            if m:
                steps.append((int(m.group(1)) + 1) / 4)  # micro-steps -> updates
                losses.append(float(m.group(2)))
        ax.plot(steps, losses, color=color, linewidth=1.5, linestyle=ls, label=label)
    ax.axvline(207, color="#777777", linewidth=0.8, linestyle=":")
    ax.text(215, 0.01, "end of warm-up", fontsize=7, color="#777777", va="bottom")
    ax.set_xlabel("update")
    ax.set_ylabel("training loss (mean of 50 steps)")
    ax.set_ylim(0, 0.56)
    ax.legend(fontsize=7, loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss_curves.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools")
    ap.add_argument("logdir")
    ap.add_argument("--out", default="docs/paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.pools)
    weight_window(d, args.out)
    inclusion_by_rank(d, args.out)
    loss_curves(args.logdir, args.out)
    print("wrote", sorted(glob.glob(os.path.join(args.out, "*.pdf"))))


if __name__ == "__main__":
    main()
