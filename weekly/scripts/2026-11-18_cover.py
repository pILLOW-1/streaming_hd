#!/usr/bin/env python3
"""Cover for the 2026-11-18 weekly report: one illustrative reasoning trajectory.

A schematic, not data: each circle is a reasoning Step; the band below shows the
Prefix state. Run from the repository root:

    python weekly/scripts/2026-11-18_cover.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("weekly/assets/2026-11-18/cover.png")
SURFACE, INK, INK2, LINE = "#fbf8f1", "#1d1832", "#5d5872", "#d9cfbd"
NEW, INHERITED, CLEAN = "#a8402f", "#e3b7a6", "#ffffff"
# Step labels and Prefix labels of the illustrative trajectory
STEP = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
PREFIX = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(-0.8, len(STEP) - 0.2)
ax.set_ylim(-1.9, 1.6)
ax.axis("off")
ax.plot([0, len(STEP) - 1], [0.4, 0.4], color=LINE, linewidth=2, zorder=1)
for t, (s, p) in enumerate(zip(STEP, PREFIX)):
    face = NEW if s else (INHERITED if p else CLEAN)
    ax.scatter(t, 0.4, s=520, color=face, edgecolor=INK if s else INK2, linewidth=1.4, zorder=3)
    ax.text(t, 0.95, f"S{t + 1}", ha="center", va="bottom", fontsize=10, color=INK2, family="DejaVu Sans Mono")
    ax.add_patch(FancyBboxPatch((t - 0.42, -0.75), 0.84, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=INHERITED if p else "#efe8da", edgecolor="none", zorder=2))
ax.text(-0.75, -0.5, "Prefix", ha="right", va="center", fontsize=10, color=INK2)
ax.text(-0.75, 0.4, "Step", ha="right", va="center", fontsize=10, color=INK2)
notes = [(2, "new error"), (4, "inherited"), (6, "recovered"), (8, "inherited")]
for t, label in notes:
    ax.annotate(label, (t, -0.8), (t, -1.45), ha="center", va="center", fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
print("wrote", OUT)
