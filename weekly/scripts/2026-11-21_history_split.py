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


# ---------------------------------------------------------------- Figure 2: Prefix alarm when an error is corrected
dyn = json.loads((REPORTS / "score_dynamics2.json").read_text())["Prefix"]["recovery"]
K = np.array(json.loads((REPORTS / "score_dynamics2.json").read_text())["offsets"])
def series(q):
    m = np.array([np.nan if v is None else v for v in dyn[q]["mean"]], float); m[np.array(dyn[q]["cells"]) < 3] = np.nan; return m
before, after = series("score1"), series("score0")      # erroneous positions before the fix, correct positions after it
keep = (K >= -5) & (K <= 10)
xb, yb = K[keep & (K < 0)], before[keep & (K < 0)]
xa, ya = K[keep & (K >= 0)], after[keep & (K >= 0)]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.axvspan(-5.5, -0.5, color="#f6e3de", zorder=0); ax.axvspan(-0.5, 10.5, color="#e4eef7", zorder=0)
ax.text(-3.0, 1.06, "error present", ha="center", fontsize=11, color=POS, fontweight="bold")
ax.text(5.0, 1.06, "error corrected", ha="center", fontsize=11, color=NEG, fontweight="bold")
ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3))); ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(10.4, 0.97, "typical error", ha="right", va="top", fontsize=9.5, color=MUTED)
ax.text(10.4, 0.03, "correct step before any error", ha="right", va="bottom", fontsize=9.5, color=MUTED)
ax.plot(np.r_[xb, xa], np.r_[yb, ya], color=INK, lw=2.6, marker="o", ms=5, zorder=3)
ax.annotate("should drop here", xy=(0, ya[0]), xytext=(1.6, 0.12), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1))
ax.set_xlim(-5.5, 10.5); ax.set_ylim(-0.08, 1.15); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("Steps relative to the correction"); ax.set_ylabel("Prefix alarm")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "score_dynamics.png", dpi=180); plt.close(fig)
