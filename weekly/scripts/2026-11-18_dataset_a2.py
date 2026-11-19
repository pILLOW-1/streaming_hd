#!/usr/bin/env python3
"""Dataset overview and high-level A2 schematic for the 2026-11-18 weekly report.

Dataset counts are computed from the frozen 6:2:2 release (labels.jsonl of every model/domain/split; 15,697
trajectories, 1,445,280 labelled steps). Run from the repository root with the dataset root as argument:

    python weekly/scripts/2026-11-18_dataset_a2.py <streaming_hd_new root>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("weekly/assets/2026-11-18")
SURFACE, RAISE, INK, INK2, LINE = "#fbf8f1", "#fffdf8", "#1a1530", "#5d5872", "#d9cfbd"
ACCENT, ACCENT_SOFT, BLUE = "#a8402f", "#f2ddd5", "#3f74b8"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
MODELS = [("qwen3_8_27b", "Qwen3.8-27B"), ("deepseek_r1_distill_llama_8b", "DeepSeek-R1-Distill-Llama-8B"),
          ("gemma_4_31b_it", "Gemma-4-31B-it")]
DOMAINS = ["Math", "Science", "Logic", "Multi-hop", "Factuality"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", OUT / name)


def counts(root):
    traj = np.zeros((3, 5), int); steps = np.zeros((3, 5), int)
    for i, (m, _) in enumerate(MODELS):
        for j, d in enumerate(DOMAINS):
            for split in ("train", "val", "test"):
                for line in open(root / m / d / split / "annotations/labels.jsonl"):
                    traj[i, j] += 1; steps[i, j] += len(json.loads(line)["labels"]["step"])
    return traj, steps


def fmt(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e4 else f"{n / 1e3:.1f}k"


def dataset(root):
    traj, steps = counts(Path(root))
    cmap = LinearSegmentedColormap.from_list("seq", ["#f7ebe4", "#e3b7a6", "#c46f58", "#8a2f22"])
    norm = LogNorm(vmin=5e3, vmax=5e5)
    fig, ax = plt.subplots(figsize=(10.2, 3.9), dpi=200)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-2.9, 6.25); ax.set_ylim(-1.05, 3.75); ax.axis("off")
    W, H, G = 1.0, 1.0, 0.06
    for j, d in enumerate(DOMAINS):
        ax.text(j + 0.5, 3.4, d, ha="center", va="center", fontsize=13, color=INK)
    ax.text(5.65, 3.4, "Total", ha="center", va="center", fontsize=13, color=INK2)
    for i, (_, name) in enumerate(MODELS):
        y = 2 - i
        ax.text(-0.15, y + 0.5, name, ha="right", va="center", fontsize=12.5, color=INK)
        for j in range(5):
            v = steps[i, j]; fc = cmap(norm(v)); dark = norm(v) > 0.55
            ax.add_patch(FancyBboxPatch((j + G, y + G), W - 2 * G, H - 2 * G, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=fc, ec="none"))
            tc = SURFACE if dark else INK
            ax.text(j + 0.5, y + 0.6, fmt(v), ha="center", va="center", fontsize=15, color=tc, weight="bold")
            ax.text(j + 0.5, y + 0.3, f"{traj[i, j]:,} traj.", ha="center", va="center", fontsize=10,
                    color=SURFACE if dark else INK2)
        ax.text(5.65, y + 0.6, fmt(steps[i].sum()), ha="center", va="center", fontsize=13, color=INK)
        ax.text(5.65, y + 0.3, f"{traj[i].sum():,} traj.", ha="center", va="center", fontsize=10, color=INK2)
    ax.plot([0.05, 4.95], [-0.02, -0.02], color=LINE, lw=1)
    for j in range(5):
        ax.text(j + 0.5, -0.3, fmt(steps[:, j].sum()), ha="center", va="center", fontsize=13, color=INK)
        ax.text(j + 0.5, -0.6, f"{traj[:, j].sum():,} traj.", ha="center", va="center", fontsize=10, color=INK2)
    ax.text(-0.15, -0.45, "Total", ha="right", va="center", fontsize=12.5, color=INK2)
    ax.text(5.65, -0.3, fmt(steps.sum()), ha="center", va="center", fontsize=13.5, color=ACCENT, weight="bold")
    ax.text(5.65, -0.6, f"{traj.sum():,} traj.", ha="center", va="center", fontsize=10, color=INK2)
    ax.text(-2.85, -0.95, "Cells: labelled steps (colour, log scale) and trajectories; each trajectory has Step, Prefix and Answer labels.",
            fontsize=10, color=INK2, va="center")
    save(fig, "dataset.png")


def node(ax, x, y, w, h, title, sub, ec, fc=RAISE, tc=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=fc, ec=ec, lw=1.6))
    ax.text(x, y + 0.17, title, ha="center", va="center", fontsize=16, color=tc)
    ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=11.5, color=INK2, style="italic")


def arrow(ax, a, b, color=INK2, lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def a2():
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.1); ax.axis("off")
    # reasoning so far: earlier steps shaded by their predicted error probability q_i, current step outlined
    tint = LinearSegmentedColormap.from_list("q", ["#f4efe5", ACCENT])
    blocks = [(0.45, "$S_1$", 0.05), (1.35, "$S_2$", 0.12), (2.25, "$S_3$", 0.85), (3.15, "$S_4$", 0.3),
              (4.55, "$S_{t-1}$", 0.1)]
    for x, lab, v in blocks:
        ax.add_patch(FancyBboxPatch((x, 0.45), 0.75, 0.55, boxstyle="round,pad=0,rounding_size=0.08", fc=tint(v), ec="none"))
        ax.text(x + 0.375, 0.725, lab, ha="center", va="center", fontsize=12, color=SURFACE if v > 0.5 else INK2)
    ax.text(4.22, 0.725, "$\\cdots$", ha="center", va="center", fontsize=14, color=INK2)
    xt = 7.22
    ax.add_patch(FancyBboxPatch((xt, 0.45), 0.75, 0.55, boxstyle="round,pad=0,rounding_size=0.08", fc=RAISE, ec=INK, lw=1.6))
    ax.text(xt + 0.375, 0.725, "$S_t$", ha="center", va="center", fontsize=13, color=INK)
    ax.text(2.86, 0.18, "earlier steps, shaded by their predicted error $q_i$", fontsize=11, color=INK2, ha="center", va="center")
    ax.text(xt + 0.375, 0.18, "current step", fontsize=11, color=INK2, ha="center", va="center")
    # error memory gathers earlier steps in proportion to q_i
    mem = (2.86, 2.2)
    for x, _, v in blocks:
        ax.add_patch(FancyArrowPatch((x + 0.375, 1.02), (mem[0], mem[1] - 0.33), arrowstyle="-", color=ACCENT,
                                     lw=0.6 + 3.2 * v, alpha=0.25 + 0.7 * v))
    ax.add_patch(FancyBboxPatch((mem[0] - 1.45, mem[1] - 0.33), 2.9, 0.66, boxstyle="round,pad=0,rounding_size=0.33",
                                fc=ACCENT_SOFT, ec=ACCENT, lw=1.4))
    ax.text(mem[0], mem[1], "error memory  $g_t=\\Sigma_{i<t}\\, q_i\\, e_i$", ha="center", va="center", fontsize=13, color=INK)
    # the two questions, one column each
    kap, stp = (2.86, 3.45), (xt + 0.375, 2.7)
    node(ax, *kap, 3.7, 0.92, "$\\kappa_t$", "is an earlier error still unresolved?", ec=ACCENT)
    node(ax, *stp, 3.2, 0.92, "$q_t$", "does this step add a new error?", ec=BLUE)
    arrow(ax, (mem[0], mem[1] + 0.34), (kap[0], kap[1] - 0.47), color=ACCENT)
    ax.text(mem[0] + 0.12, 2.76, "+ prefix context", fontsize=10.5, color=INK2, va="center")
    arrow(ax, (stp[0], 1.02), (stp[0], stp[1] - 0.47), color=BLUE)
    # composition
    out = (5.3, 4.62)
    ax.add_patch(FancyBboxPatch((out[0] - 2.0, out[1] - 0.3), 4.0, 0.6, boxstyle="round,pad=0,rounding_size=0.3", fc=INK, ec=INK))
    ax.text(*out, "Prefix  $\\pi^{\\mathrm{P}}_t = q_t + (1-q_t)\\,\\kappa_t$", ha="center", va="center", fontsize=14.5, color=SURFACE)
    arrow(ax, (kap[0], kap[1] + 0.47), (out[0] - 1.6, out[1] - 0.31), color=ACCENT, rad=-0.25)
    arrow(ax, (stp[0], stp[1] + 0.47), (out[0] + 1.6, out[1] - 0.31), color=BLUE, rad=0.25)
    save(fig, "a2_framework.png")


if __name__ == "__main__":
    dataset(sys.argv[1])
    a2()
