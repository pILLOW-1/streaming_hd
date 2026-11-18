#!/usr/bin/env python3
"""Figures for the 2026-11-18 weekly report (schematics and summary charts), in a paper-style serif.

Numbers in the charts are copied from published results in this repository:
- Step TEST AUROC on Qwen3.8-27B: experiments/{saplma,hami}/622_p70_20261108/result/qwen3_8_27b/<domain>/step.json
  and experiments/streaming_622_p70/result/metrics.json (HaMI has no Qwen Math/Science result yet).
- Prefix composition: experiments/step_prefix_composition_20261111 (84.74% of Prefix positives inherited).
- Recovery: experiments/prefix_recovery_diagnostic_20261112 (TEST means over 71 detector-combination pairs).
Run from the repository root:  python weekly/scripts/2026-11-18_figures.py [optional still-frame path]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("weekly/assets/2026-11-18")
SURFACE, RAISE, INK, INK2, LINE = "#fbf8f1", "#fffdf8", "#1a1530", "#5d5872", "#d9cfbd"
NEW, INHERITED, CLEAN_FILL, INH_TEXT = "#a8402f", "#e3b7a6", "#efe8da", "#b0664f"
STREAMING, SAPLMA, HAMI = "#3f74b8", "#d95926", "#199e70"  # validated on SURFACE
SERIF = "STIXGeneral"  # Times-like, as in ICML papers; math uses the matching STIX fonts
plt.rcParams.update({"font.family": SERIF, "font.size": 12, "mathtext.fontset": "stix",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def canvas(w, h, xlim, ylim, dpi=200):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=RAISE, ec=INK2, lw=1.1, fs=13, color=INK, r=0.08, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, **kw)


def arrow(ax, a, b, color=INK2, lw=1.2, ms=13):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def save(fig, name, pad=0.06):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print("wrote", OUT / name)


def style_axis(ax):
    ax.tick_params(colors=INK2, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)


# ---------------------------------------------------------------- 1. task: three levels on one trajectory
def background():
    # S_1 ... S_5, an ellipsis, then S_T.
    cols = [0, 1, 2, 3, 4, 6]  # x positions; 5 is the ellipsis
    names = [r"$\mathbf{S}_1$", r"$\mathbf{S}_2$", r"$\mathbf{S}_3$", r"$\mathbf{S}_4$", r"$\mathbf{S}_5$", r"$\mathbf{S}_T$"]
    step = [0, 0, 1, 0, 0, 0]
    prefix = [0, 0, 1, 1, 0, 1]
    fig, ax = canvas(8.2, 3.05, (-3.35, 9.0), (-1.95, 2.05))
    box(ax, -1.65, -0.62, 0.9, 1.92, r"$\mathbf{Q}$", fs=20)
    ax.text(-1.2, -0.92, "question", ha="center", fontsize=12, color=INK2)
    ax.plot([0, 4.3], [0.85, 0.85], color=LINE, lw=2, zorder=1)
    ax.plot([5.7, 6], [0.85, 0.85], color=LINE, lw=2, zorder=1)
    ax.text(5.0, 0.85, r"$\cdots$", ha="center", va="center", fontsize=20, color=INK2)
    ax.text(5.0, -0.25, r"$\cdots$", ha="center", va="center", fontsize=20, color=INK2)
    for x, nm, s, p in zip(cols, names, step, prefix):
        face = NEW if s else (INHERITED if p else "white")
        ax.scatter(x, 0.85, s=380, color=face, edgecolor=INK if s else INK2, lw=1.2, zorder=3)
        ax.text(x, 1.3, nm, ha="center", fontsize=15, color=INK2)
        ax.add_patch(FancyBboxPatch((x - 0.38, -0.52), 0.76, 0.55, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=INHERITED if p else CLEAN_FILL, ec="none", zorder=2))
    box(ax, 6.85, -0.52, 1.2, 1.62, r"$\mathbf{A}$", fc=NEW, ec=INK, fs=20, color="white")
    ax.text(7.45, 1.3, r"$y^{\mathrm{A}}$", ha="center", fontsize=16, color=INK)
    ax.text(7.45, -0.82, "answer", ha="center", fontsize=12, color=INK2)
    ax.text(-3.3, 0.85, r"Step $y^{\mathrm{S}}_t$", ha="left", va="center", fontsize=14, color=INK)
    ax.text(-3.3, -0.25, r"Prefix $y^{\mathrm{P}}_t$", ha="left", va="center", fontsize=14, color=INK)
    ax.plot([-1.65, 3.35], [1.82, 1.82], color=INK, lw=1.1)
    for xx in (-1.65, 3.35):
        ax.plot([xx, xx], [1.7, 1.82], color=INK, lw=1.1)
    ax.text(0.85, 1.88, r"at step $t$, detectors read $(\mathbf{Q},\mathbf{S}_{1:t})$", ha="center", va="bottom",
            fontsize=13, color=INK)
    ax.plot([-0.4, 6.4], [-1.12, -1.12], color=INK2, lw=0.9)
    ax.text(3.0, -1.2, r"reasoning $\mathbf{R}=\mathbf{S}_{1:T}$", ha="center", va="top", fontsize=13, color=INK2)
    for lx, (fc, ec, lab) in zip([0.0, 2.3, 5.15], [(NEW, INK, "new error"), (INHERITED, INK2, "unresolved error"),
                                                   ("white", INK2, "clean")]):
        ax.scatter(lx, -1.78, s=120, color=fc, edgecolor=ec, lw=1)
        ax.text(lx + 0.25, -1.78, lab, va="center", fontsize=12.5, color=INK)
    save(fig, "background.png")


# ---------------------------------------------------------------- 2. decomposition (whitespace allowed)
def decomposition():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 10); ax.set_ylim(-0.15, 4.0); ax.axis("off")
    fs = 36
    parts = [(r"$\pi^{\mathrm{P}}_t\;=\;$", INK), (r"$\pi^{\mathrm{S}}_t$", NEW), (r"$\;+\;$", INK),
             (r"$(1-\pi^{\mathrm{S}}_t)\,\kappa_t$", INH_TEXT)]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widths, texts = [], []
    for s, c in parts:
        tt = ax.text(0, 2.95, s, fontsize=fs, color=c, va="center")
        widths.append(tt.get_window_extent(renderer=r).transformed(ax.transData.inverted()).width); texts.append(tt)
    x = (10 - sum(widths)) / 2
    spans = []
    for tt, w in zip(texts, widths):
        tt.set_x(x); spans.append((x, x + w)); x += w
    for i, lab, c in [(1, "whether the error\nis new", NEW), (3, "whether the error\nis inherited", INH_TEXT)]:
        a, b = spans[i]
        ax.plot([a + 0.05, a + 0.05, b - 0.05, b - 0.05], [2.42, 2.3, 2.3, 2.42], color=c, lw=1.5)
        mid = (a + b) / 2
        arrow(ax, (mid, 2.28), (mid, 1.72), color=c, lw=1.5, ms=16)
        ax.text(mid, 1.62, lab, ha="center", va="top", fontsize=17, color=c, fontweight="bold", linespacing=1.15)
    ax.add_patch(FancyBboxPatch((1.4, -0.08), 7.2, 0.8, boxstyle="round,pad=0,rounding_size=0.12", fc=RAISE, ec=LINE, lw=1.1))
    ax.text(5, 0.32, r"$\kappa_t=\mathbb{P}_{D^{\mathrm{P}}}\left(y^{\mathrm{P}}_t=1\;\middle|\;y^{\mathrm{S}}_t=0,\ "
            r"\mathbf{Q},\mathbf{S}_{1:t}\right)$", ha="center", va="center", fontsize=19, color=INK)
    save(fig, "decomposition.png", pad=0.04)


# ---------------------------------------------------------------- 3a. step read-outs (animated) + AUROC by domain
STEP_AUROC = {  # Qwen3.8-27B Step TEST AUROC; None = not run yet
    "Streaming": [0.8716, 0.8720, 0.8145, 0.7852, 0.8953],
    "SAPLMA": [0.8822, 0.8607, 0.8255, 0.8046, 0.9022],
    "HaMI": [0.8875, 0.8735, None, None, 0.9080],
}
DOMAINS = ["Logic", "Factuality", "Science", "Math", "Multi-hop"]


def step_readouts(still=None):
    n = 12
    idx = np.arange(n)
    readouts = [
        ("SAPLMA", "last token", SAPLMA, np.eye(n)[-1]),
        ("Streaming", "fixed weights,\nrising to the end", STREAMING, np.exp(3 * idx / (n - 1))),
        ("HaMI", "learned weights on\nselected tokens", HAMI, np.array([.02, .03, .02, .2, .05, .02, .3, .04, .02, .2, .05, .05])),
    ]
    for _, _, _, w in readouts:
        w /= w.max()
    fig = plt.figure(figsize=(11, 3.7), dpi=110)
    axl = fig.add_axes([0.0, 0.0, 0.55, 1.0]); axl.axis("off"); axl.set_xlim(-4.6, n + 2.0); axl.set_ylim(-0.75, 7.4)
    axr = fig.add_axes([0.62, 0.15, 0.37, 0.7])
    rows_y = [5.4, 3.0, 0.6]
    tokens, bars, outs = [], [], []
    for (name, desc, col, w), y in zip(readouts, rows_y):
        axl.text(-4.5, y + 0.55, name, fontsize=15, color=col, fontweight="bold", va="center")
        axl.text(-4.5, y + 0.15, desc, fontsize=12, color=INK2, va="top", linespacing=1.1)
        row_t, row_b = [], []
        for i in range(n):
            p = Rectangle((i - 0.38, y - 0.32), 0.76, 0.56, fc=CLEAN_FILL, ec=INK2, lw=0.8, alpha=0)
            axl.add_patch(p); row_t.append(p)
            b = Rectangle((i - 0.3, y + 0.3), 0.6, 0.0, fc=col, ec="none")
            axl.add_patch(b); row_b.append(b)
        o = FancyBboxPatch((n + 0.3, y - 0.32), 1.4, 0.56, boxstyle="round,pad=0,rounding_size=0.1", fc=col, ec="none", alpha=0)
        axl.add_patch(o)
        ot = axl.text(n + 1.0, y - 0.04, r"$\mathbf{h}_t$", ha="center", va="center", color="white", fontsize=16, alpha=0)
        tokens.append(row_t); bars.append(row_b); outs.append((o, ot))
    axl.text(n / 2 - 0.5, -0.65, "tokens of step $t$", ha="center", fontsize=12.5, color=INK2)
    x = np.arange(len(DOMAINS)); wbar = 0.26
    for k, (name, col) in enumerate([("Streaming", STREAMING), ("SAPLMA", SAPLMA), ("HaMI", HAMI)]):
        for i, v in enumerate(STEP_AUROC[name]):
            pos = x[i] + (k - 1) * wbar
            if v is None:
                axr.text(pos, 0.753, "n/a", ha="center", va="bottom", fontsize=10, color=INK2, rotation=90)
                continue
            axr.bar(pos, v - 0.75, wbar - 0.03, bottom=0.75, color=col, edgecolor=SURFACE, lw=0.8, zorder=3,
                    label=name if i == 0 else None)
    for i in range(len(DOMAINS)):
        vals = {k: v[i] for k, v in STEP_AUROC.items() if v[i] is not None}
        best = max(vals, key=vals.get)
        k = ["Streaming", "SAPLMA", "HaMI"].index(best)
        axr.text(x[i] + (k - 1) * wbar, vals[best] + 0.002, r"$\bigstar$", ha="center", va="bottom", fontsize=11, color=INK)
    axr.set_xticks(x, DOMAINS, fontsize=12.5, color=INK); axr.set_ylim(0.75, 0.925)
    axr.set_yticks([0.75, 0.80, 0.85, 0.90]); axr.tick_params(axis="y", labelsize=11.5)
    axr.set_ylabel("Step AUROC", fontsize=13, color=INK2)
    axr.grid(axis="y", color="#e6dfd0", lw=0.8, zorder=0); style_axis(axr)
    axr.legend(ncol=3, frameon=False, fontsize=12.5, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.2,
               columnspacing=1.2)

    frames_tok, frames_w, frames_hold = n, 14, 14

    def update(f):
        shown = min(f + 1, n)
        grow = np.clip((f - frames_tok) / (frames_w - 4), 0, 1)
        for r, (row_t, row_b) in enumerate(zip(tokens, bars)):
            w = readouts[r][3]
            for i in range(n):
                row_t[i].set_alpha(1.0 if i < shown else 0.0)
                row_b[i].set_height(1.05 * w[i] * grow)
            outs[r][0].set_alpha(1.0 if grow >= 1 else 0.0)
            outs[r][1].set_alpha(1.0 if grow >= 1 else 0.0)
        return []

    anim = FuncAnimation(fig, update, frames=frames_tok + frames_w + frames_hold, interval=120)
    anim.save(OUT / "step_readouts.gif", writer=PillowWriter(fps=8))
    if still:
        update(frames_tok + frames_w + frames_hold - 1)
        fig.savefig(still, dpi=110)
    plt.close(fig)
    print("wrote", OUT / "step_readouts.gif")


# ---------------------------------------------------------------- 3b. prefix challenge
def prefix_challenge():
    fig = plt.figure(figsize=(8.6, 2.75), dpi=200)
    a = fig.add_axes([0.0, 0.0, 0.36, 0.84]); b = fig.add_axes([0.55, 0.22, 0.43, 0.6])
    # (a) composition of Prefix-erroneous positions; the small "new" wedge faces right
    a.pie([15.3, 84.7], colors=[NEW, INHERITED], startangle=-27.5, counterclock=True, radius=1.0,
          wedgeprops=dict(edgecolor=SURFACE, linewidth=2))
    a.set_xlim(-1.1, 2.2); a.set_ylim(-1.08, 1.08); a.set_aspect("equal"); a.axis("off")
    a.text(-0.5, 0.0, "inherited\n85%", ha="center", va="center", fontsize=15, color=INK, fontweight="bold", linespacing=1.15)
    a.annotate("new  15%", xy=(0.98, 0.0), xytext=(1.25, 0.0), fontsize=15, color=INK, va="center", fontweight="bold",
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    fig.text(0.0, 0.985, "(a) Where Prefix errors come from", fontsize=14, color=INK, va="top")
    # (b) separating inherited errors from two kinds of clean positions
    vals, cols = [0.933, 0.661], ["#8a94a8", NEW]
    for i, (v, c) in enumerate(zip(vals, cols)):
        b.barh(i, v - 0.5, left=0.5, color=c, height=0.56, zorder=3)
        b.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=14, color=INK, fontweight="bold")
    b.set_yticks([0, 1], ["vs. fresh clean", "vs. recovered clean"], fontsize=13.5, color=INK); b.invert_yaxis()
    b.set_xlim(0.5, 1.0); b.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); b.tick_params(axis="x", labelsize=11.5)
    b.set_xlabel("AUROC of existing detectors (0.5 = chance)", fontsize=12.5, color=INK2)
    b.grid(axis="x", color="#e6dfd0", lw=0.8, zorder=0); style_axis(b)
    fig.text(0.37, 0.985, "(b) Telling inherited errors from clean positions", fontsize=14, color=INK, va="top")
    save(fig, "prefix_challenge.png")


# ---------------------------------------------------------------- 4b. method: kappa on Step-clean positions
def prefix_method():
    fig, ax = canvas(9.0, 2.05, (0, 11.3), (0.15, 3.0))
    step = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    prefix = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    for t in range(10):
        used = step[t] == 0
        x = 0.3 + t * 0.42
        ax.scatter(x, 2.35, s=150, color=NEW if step[t] else (INHERITED if prefix[t] else "white"),
                   edgecolor=INK2, lw=1, alpha=1 if used else 0.25, zorder=3)
        if used:
            ax.add_patch(Rectangle((x - 0.17, 1.85), 0.34, 0.2, fc=INHERITED if prefix[t] else CLEAN_FILL, ec="none"))
        else:
            ax.text(x, 2.35, "×", ha="center", va="center", fontsize=14, color=INK, zorder=4)
    ax.text(2.2, 1.45, r"only positions with $y^{\mathrm{S}}_t=0$", ha="center", fontsize=13.5, color=INK)
    ax.text(2.2, 1.05, "inherited vs. recovered", ha="center", fontsize=12.5, color=INK2)
    arrow(ax, (4.4, 2.1), (4.95, 2.1))
    box(ax, 5.0, 1.68, 2.2, 0.85, "different read-outs", fs=13.5)
    ax.text(6.1, 1.6, "e.g. SAPLMA / Streaming / HaMI", ha="center", va="top", fontsize=11.5, color=INK2)
    arrow(ax, (7.25, 2.1), (7.8, 2.1))
    ax.text(8.15, 2.1, r"$\kappa_t$", ha="center", va="center", fontsize=22, color=INH_TEXT)
    arrow(ax, (8.15, 1.75), (8.15, 1.1))
    box(ax, 7.25, 0.25, 3.95, 0.8, r"$\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t+(1-\pi^{\mathrm{S}}_t)\,\kappa_t$", fs=15)
    ax.text(10.1, 1.75, r"frozen Step $\pi^{\mathrm{S}}_t$", ha="center", fontsize=12.5, color=INK2)
    arrow(ax, (10.1, 1.62), (10.1, 1.1))
    save(fig, "prefix_method.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    background(); decomposition(); step_readouts(sys.argv[1] if len(sys.argv) > 1 else None)
    prefix_challenge(); prefix_method()
