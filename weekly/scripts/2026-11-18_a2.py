#!/usr/bin/env python3
"""A2 comparison figures for the 2026-11-18 weekly report (the schematic is in 2026-11-18_dataset_a2.py).

Numbers are copied from published TEST results in this repository (Qwen3.8-27B, single seed 42, VAL-selected layer):
- A1 and A2 (fixed readouts): experiments/qwen_c18_a2_series_20261115/README.md (Frozen TEST comparison) and
  experiments/qwen_c18_error_memory_20261114/README.md (pooled TEST Prefix recovery AUROC).
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
STREAMING, BASELINE, ABLATION, OURS = "#3f74b8", "#b08637", "#8a6fb8", "#a8402f"  # validated on SURFACE
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


SERIES = [("Streaming", STREAMING, ""), ("Best baseline", BASELINE, "///"),
          ("Ours (w/o error memory)", ABLATION, ""), ("Ours", OURS, "")]


def bars(ax, x, values, lo, hi, best_name, domains):
    w = 0.2
    for k, ((name, color, hatch), vals) in enumerate(zip(SERIES, values)):
        pos = x + (k - 1.5) * w
        ax.bar(pos, np.array(vals) - lo, w - 0.02, bottom=lo, color=color, hatch=hatch, edgecolor=SURFACE,
               linewidth=1.2, label=name, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + (hi - lo) * 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    color=INK if name == "Ours" else INK2)
    ax.set_xticks(x, [f"{d}\n(best: {b})" for d, b in zip(domains, best_name)], color=INK)
    ax.set_ylim(lo, hi)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color="#e6dfd0", linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)


def results():
    domains = ["Logic", "Factuality"]; x = np.arange(2)
    panels = [  # Streaming, best baseline, A1, A2
        ("Step", [[0.8716, 0.8720], [0.8875, 0.8735], [0.8775, 0.8820], [0.8775, 0.8826]], ["HaMI", "HaMI"], (0.85, 0.895)),
        ("Prefix", [[0.8256, 0.8222], [0.8256, 0.8226], [0.8341, 0.8106], [0.8377, 0.8367]], ["Streaming", "Global Mean"], (0.80, 0.845)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=200)
    for ax, (title, values, best_name, (lo, hi)) in zip(axes, panels):
        bars(ax, x, values, lo, hi, best_name, domains)
        ax.set_title(f"{title} TEST AUROC", fontsize=13, color=INK, pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07), labelcolor=INK)
    fig.tight_layout()
    save(fig, "a2_results.png")


def recovery():
    domains = ["Logic", "Factuality"]; x = np.arange(2)
    values = [[0.7020, 0.7205], [0.7020, 0.7233], [0.7073, 0.6895], [0.7204, 0.7476]]
    fig, ax = plt.subplots(figsize=(10.0, 3.6), dpi=200)
    bars(ax, x, values, 0.65, 0.76, ["Streaming", "Global Mean"], domains)
    ax.set_title("Prefix Recovery TEST AUROC", fontsize=13, color=INK, pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 1.0), labelcolor=INK)
    save(fig, "a2_recovery.png")


INH_TEXT = "#b0664f"

if __name__ == "__main__":
    results()
    recovery()
