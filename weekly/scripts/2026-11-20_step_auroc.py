#!/usr/bin/env python3
"""Figure for the 2026-11-20 weekly report: Qwen Step TEST AUROC by domain.

Values are copied from experiments/qwen_step_only_learnable_20261115/README.md
(legacy-recipe table): Original Streaming, best published baseline per domain,
and single-layer Learnable aggregation. Run from the repository root:

    python weekly/scripts/2026-11-20_step_auroc.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("weekly/assets/2026-11-20/step_auroc.png")
DOMAINS = ["Logic", "Factuality", "Science", "Math", "Multi-hop"]
SERIES = [  # name, values, color (validated palette on the page surface), hatch
    ("Streaming", [0.8716, 0.8720, 0.8145, 0.7852, 0.8953], "#3f74b8", ""),
    ("Best baseline", [0.8875, 0.8735, 0.8255, 0.8046, 0.9080], "#b08637", "///"),
    ("Learnable aggregation", [0.8871, 0.8816, 0.8219, 0.7932, 0.9177], "#a8402f", ""),
]
BEST = ["HaMI", "HaMI", "SAPLMA", "SAPLMA", "HaMI"]
SURFACE, INK, INK2, GRID = "#fbf8f1", "#1d1832", "#5d5872", "#e6dfd0"

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12, "hatch.linewidth": 0.8,
                     "hatch.color": SURFACE})
fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x = np.arange(len(DOMAINS))
width = 0.26
for k, (name, vals, color, hatch) in enumerate(SERIES):
    pos = x + (k - 1) * width
    ax.bar(pos, np.array(vals) - 0.75, width - 0.02, bottom=0.75, color=color, hatch=hatch,
           edgecolor=SURFACE, linewidth=1.2, label=name, zorder=3)
    if name == "Learnable aggregation":  # selective direct labels: our method only
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x, [f"{d}\n(best: {b})" for d, b in zip(DOMAINS, BEST)], color=INK)
ax.set_ylim(0.75, 0.94)
ax.set_yticks(np.arange(0.75, 0.951, 0.05))
ax.set_ylabel("Step TEST AUROC (axis starts at 0.75)", color=INK2)
ax.tick_params(axis="both", colors=INK2, length=0)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK2)
ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13), labelcolor=INK)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.08)
print("wrote", OUT)
