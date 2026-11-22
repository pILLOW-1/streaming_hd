#!/usr/bin/env python3
"""Figures for the 2026-11-21 weekly report.

  weekly/assets/2026-11-21/history_split.png   A_pre vs A_post grouped bars, Step and Prefix, three baselines + CHAINED
  weekly/assets/2026-11-21/score_dynamics.png  Streaming score percentile around the first error / first recovery

Reads the local diagnostic outputs of the random 6:2:2 split (TEST):
  python weekly/scripts/2026-11-21_history_split.py <diagnostic reports dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "assets/2026-11-21"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, SOFT = "#2b2b2b", "#8a8a85", "#ececea", "#d8d8d4"
POS, NEG = "#a8402f", "#3f74b8"
METHODS = {"SAPLMA": "#2a9d8f", "Streaming": "#3f74b8", "Global Mean": "#b08637"}
# Streaming on CHAINED (independent generators, annotator and Step definition), provided results.
CHAINED = {"Step": [(0.9337, 0.7495), (0.9621, 0.8862)], "Prefix": [(0.9010, 0.6824), (0.9085, 0.8108)]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c8c8c4",
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})

rows = json.loads((REPORTS / "history_split_all.json").read_text())


# ---------------------------------------------------------------- Figure 1: grouped bars
PRE, POST = "#9db9df", POS   # light blue = negatives before any error, red = negatives after an error
names = list(METHODS) + ["CHAINED"]
labels = ["SAPLMA", "Streaming", "Global\nMean", "Streaming\n(CHAINED)"]
xs = np.array([0.0, 1.0, 2.0, 3.35]); w = 0.36
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
for ax, (level, key) in zip(axes, (("Step", "step"), ("Prefix", "prefix"))):
    ax.axvspan(2.75, 3.95, color="#f4f3ef", zorder=0)
    for x0, name in zip(xs, names):
        pairs = np.array(CHAINED[level] if name == "CHAINED" else
                         [(r[key]["A_pre"], r[key]["A_post"]) for r in rows if r["method"] == name])
        mean = pairs.mean(0)
        for j, (dx, color) in enumerate(((-w / 2, PRE), (w / 2, POST))):
            lo, hi = np.percentile(pairs[:, j], [25, 75]) if len(pairs) > 2 else (pairs[:, j].min(), pairs[:, j].max())
            ax.bar(x0 + dx, mean[j] - 0.5, w * 0.92, bottom=0.5, color=color, zorder=2)
            ax.plot([x0 + dx] * 2, [lo, hi], color=INK, lw=1, alpha=0.55, zorder=3)
            ax.text(x0 + dx, max(mean[j], hi) + 0.012, f"{mean[j]:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(xs, labels); ax.tick_params(axis="x", length=0, labelsize=10.5, labelcolor=INK)
    ax.set_ylim(0.5, 1.04); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(level, color=INK, fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("TEST AUROC")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=PRE, label="Errors vs. correct positions before any error"), Patch(color=POST, label="Errors vs. correct positions after an earlier error")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
fig.savefig(OUT / "history_split.png", dpi=180)
plt.close(fig)


# ---------------------------------------------------------------- Figure 2: label vs detector around events
dyn = json.loads((REPORTS / "score_dynamics2.json").read_text())
K = np.array(dyn["offsets"])
LABEL, DET = "#2b2b2b", "#a8402f"
fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True, sharey=True)
for r_i, level in enumerate(("Step", "Prefix")):
    for c_i, event in enumerate(("error", "recovery")):
        ax = axes[r_i, c_i]; d = dyn[level][event]
        for y0 in (0, 1):
            ax.axhline(y0, color=SOFT, lw=1, zorder=0)
        ax.axvline(0, color=SOFT, lw=1, ls=(0, (3, 3)), zorder=0)
        def series(q):
            m = np.array([np.nan if v is None else v for v in d[q]["mean"]], float)
            m[np.array(d[q]["cells"]) < 3] = np.nan
            return m
        ax.step(K, series("label"), where="mid", color=LABEL, lw=1.8, label="Ground truth: share labelled erroneous", zorder=3)
        ax.plot(K, series("score"), color=DET, lw=2.4, marker="o", ms=3.5, label="Detector: how error-like the position looks", zorder=4)
        ax.set_ylim(-0.12, 1.12); ax.set_xticks(range(-5, 16, 5)); ax.set_yticks([0, 0.5, 1])
        ax.spines[["top", "right"]].set_visible(False)
        if r_i == 0:
            ax.set_title("Around the first error" if event == "error" else "Around the first recovery", color=INK, fontsize=11.5)
        if c_i == 0:
            ax.set_ylabel(level, fontsize=11.5)
        if r_i == 1:
            ax.set_xlabel("Steps relative to the event")
axes[0, 0].set_yticklabels(["0", "0.5", "1"])
axes[1, 1].annotate("label drops to 0,\ndetector does not", xy=(0, 0.37), xytext=(3.2, 0.9), fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "score_dynamics.png", dpi=180)
plt.close(fig)
