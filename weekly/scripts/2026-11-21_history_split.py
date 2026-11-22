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


# ---------------------------------------------------------------- Figure 2: retraining on post-error positions
# Qwen Multi-hop, random 6:2:2 TEST, AUROC on post-error positions. Streaming: accepted standard predictions vs. the
# same recipe trained on post-error positions only (Step L51, Prefix L64). Probes: logistic / MLP on the same pooled
# features and layers, trained on all vs. post-error TRAIN positions.
CHECK = Path("/shared/homes/u26487468/agent_uncertainty_quantification/experiments/streaming_hd/runs/"
             "qwen_a2_memory_forgetting_check_20261119/reports")
probe = json.loads((CHECK / "entanglement_random/result.json").read_text())
stream = json.loads((CHECK / "entanglement_baselines/result.json").read_text())
ALLC, POSTC = "#c9c9c4", "#2a9d8f"
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
for ax, level in zip(axes, ("Step", "Prefix")):
    st = stream[f"Streaming/{level}@L{51 if level == 'Step' else 64}"]
    groups = [("Streaming", st["standard"]["within_post"], st["post_only_within_post"], st["standard"]["A_pre"])]
    for kind, name in (("logistic", "Linear probe"), ("mlp", "MLP probe")):
        t = probe[level][kind]["TEST"]
        groups.append((name, t["all"]["within_post"], t["post_within_post"], t["all"]["A_pre"]))
    xs = np.arange(len(groups)); w = 0.34
    for x0, (name, a_all, a_post, _) in zip(xs, groups):
        for dx, v, color in ((-w / 2, a_all, ALLC), (w / 2, a_post, POSTC)):
            ax.bar(x0 + dx, v - 0.5, w * 0.92, bottom=0.5, color=color, zorder=2)
            ax.text(x0 + dx, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ref = np.mean([g[3] for g in groups])
    ax.axhline(ref, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(len(groups) - 0.55, ref + 0.012, "before any error", ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(xs, [g[0] for g in groups]); ax.tick_params(axis="x", length=0, labelcolor=INK, labelsize=10.5)
    ax.set_ylim(0.5, 1.04); ax.set_title(level, color=INK, fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("AUROC after an error")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=ALLC, label="Trained on all positions"), Patch(color=POSTC, label="Trained on post-error positions only")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
fig.savefig(OUT / "retrain_post_error.png", dpi=180)
plt.close(fig)
