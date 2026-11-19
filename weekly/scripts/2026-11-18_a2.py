#!/usr/bin/env python3
"""A2 Step/Prefix comparison figure for the 2026-11-18 weekly report (the schematic is in 2026-11-18_dataset_a2.py).

Numbers are copied from published TEST results in this repository (Qwen3.8-27B, single seed 42, VAL-selected layer):
- A2 fixed: experiments/qwen_c18_a2_series_20261115/README.md (Frozen TEST comparison).
- Streaming and best baseline: experiments/qwen_step_only_learnable_20261115/README.md (Step) and
  experiments/qwen_baseline_prefix_decomposition_20261116/README.md (Prefix).
Run from the repository root:  python weekly/scripts/2026-11-18_a2.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("weekly/assets/2026-11-18")
SURFACE, RAISE, INK, INK2, LINE = "#fbf8f1", "#fffdf8", "#1a1530", "#5d5872", "#d9cfbd"
NEW, INHERITED, CLEAN_FILL = "#a8402f", "#e3b7a6", "#efe8da"
STREAMING, BASELINE, OURS = "#3f74b8", "#b08637", "#a8402f"
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 12, "mathtext.fontset": "stix", "hatch.linewidth": 0.8,
                     "hatch.color": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def box(ax, x, y, w, h, text="", fc=RAISE, ec=INK2, lw=1.1, fs=12.5, color=INK, r=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.35)


def arrow(ax, a, b, color=INK2, lw=1.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", OUT / name)


def results():
    domains = ["Logic", "Factuality"]
    panels = [
        ("Step", [0.8716, 0.8720], [0.8875, 0.8735], ["HaMI", "HaMI"], [0.8775, 0.8826], (0.85, 0.895)),
        ("Prefix", [0.8256, 0.8222], [0.8256, 0.8226], ["Streaming", "Global Mean"], [0.8377, 0.8367], (0.80, 0.845)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), dpi=200)
    x = np.arange(len(domains)); w = 0.26
    for ax, (title, stream, best, best_name, a2, (lo, hi)) in zip(axes, panels):
        for k, (name, vals, color, hatch) in enumerate([("Streaming", stream, STREAMING, ""),
                                                        ("Best baseline", best, BASELINE, "///"),
                                                        ("A2", a2, OURS, "")]):
            pos = x + (k - 1) * w
            ax.bar(pos, np.array(vals) - lo, w - 0.02, bottom=lo, color=color, hatch=hatch, edgecolor=SURFACE,
                   linewidth=1.2, label=name, zorder=3)
            for p, v in zip(pos, vals):
                ax.text(p, v + (hi - lo) * 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                        color=INK if name == "A2" else INK2)
        ax.set_xticks(x, [f"{d}\n(best: {b})" for d, b in zip(domains, best_name)], color=INK)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{title} TEST AUROC", fontsize=13, color=INK, pad=8)
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="y", color="#e6dfd0", linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
    axes[0].set_ylabel("axis does not start at 0", color=INK2, fontsize=10.5)
    axes[0].legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.3), labelcolor=INK)
    fig.tight_layout()
    save(fig, "a2_results.png")


INH_TEXT = "#b0664f"

if __name__ == "__main__":
    results()
