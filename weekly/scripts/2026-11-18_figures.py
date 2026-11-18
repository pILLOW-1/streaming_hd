#!/usr/bin/env python3
"""Figures for the 2026-11-18 weekly report (schematics and summary charts).

Numbers in the charts are copied from published results in this repository:
- Step TEST AUROC on Qwen3.8-27B: experiments/{saplma,hami}/622_p70_20261108/result/qwen3_8_27b/<domain>/step.json
  and experiments/streaming_622_p70/result/metrics.json (HaMI has no Qwen Math/Science result yet).
- Prefix composition: experiments/step_prefix_composition_20261111 (84.74% of Prefix positives inherited).
- Recovery: experiments/prefix_recovery_diagnostic_20261112 (TEST means over 71 detector-combination pairs).
Run from the repository root:  python weekly/scripts/2026-11-18_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("weekly/assets/2026-11-18")
SURFACE, RAISE, INK, INK2, LINE = "#fbf8f1", "#fffdf8", "#1a1530", "#5d5872", "#d9cfbd"
NEW, INHERITED, CLEAN_FILL = "#a8402f", "#e3b7a6", "#efe8da"
STREAMING, SAPLMA, HAMI = "#3f74b8", "#d95926", "#199e70"  # validated on SURFACE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "mathtext.fontset": "stix",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=RAISE, ec=INK2, lw=1.2, fs=11, color=INK, r=0.08, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, **kw)


def arrow(ax, a, b, color=INK2, lw=1.3, style="-|>", ms=12):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw))


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. task: three levels on one trajectory
def background():
    step = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    prefix = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    T = len(step)
    fig, ax = canvas(10.5, 4.6, (-4.1, T + 2.2), (-2.75, 2.35))
    box(ax, -1.75, -0.75, 1.0, 2.15, r"$\mathbf{Q}$", fc=RAISE, fs=18)
    ax.text(-1.25, -1.1, "question", ha="center", fontsize=9.5, color=INK2)
    ax.plot([0, T - 1], [0.9, 0.9], color=LINE, lw=2, zorder=1)
    for t in range(T):
        face = NEW if step[t] else (INHERITED if prefix[t] else "white")
        ax.scatter(t, 0.9, s=430, color=face, edgecolor=INK if step[t] else INK2, lw=1.3, zorder=3)
        ax.text(t, 1.42, rf"$\mathbf{{S}}_{{{t + 1}}}$", ha="center", fontsize=13, color=INK2)
        ax.add_patch(FancyBboxPatch((t - 0.4, -0.55), 0.8, 0.6, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=INHERITED if prefix[t] else CLEAN_FILL, ec="none", zorder=2))
    box(ax, T - 0.1, -0.55, 1.35, 1.75, r"$\mathbf{A}$", fc=NEW, ec=INK, fs=18, color="white")
    ax.text(T + 0.575, -0.9, "answer", ha="center", fontsize=9.5, color=INK2)
    ax.text(-4.05, 0.9, r"Step  $y^{\mathrm{S}}_t$", ha="left", va="center", fontsize=12, color=INK)
    ax.text(-4.05, -0.25, r"Prefix  $y^{\mathrm{P}}_t$", ha="left", va="center", fontsize=12, color=INK)
    ax.text(T + 0.575, 1.55, r"Answer  $y^{\mathrm{A}}$", ha="center", fontsize=12, color=INK)
    # what a streaming detector reads at step t = 5
    t = 4
    ax.plot([-1.75, t + 0.4], [2.05, 2.05], color=INK, lw=1.2)
    for xx in (-1.75, t + 0.4):
        ax.plot([xx, xx], [1.92, 2.05], color=INK, lw=1.2)
    ax.text((t - 1.35) / 2, 2.13, r"at step $t$ the detectors read $(\mathbf{Q},\mathbf{S}_{1:t})$", ha="center",
            va="bottom", fontsize=11, color=INK)
    ax.plot([0 - 0.45, T - 1 + 0.45], [-1.75, -1.75], color=INK2, lw=1)
    ax.text((T - 1) / 2, -1.95, r"reasoning $\mathbf{R}=\mathbf{S}_{1:T}$", ha="center", va="top", fontsize=11, color=INK2)
    # legend
    for k, (fc, ec, lab) in enumerate([(NEW, INK, "new error"), (INHERITED, INK2, "earlier error still unresolved"),
                                       ("white", INK2, "no unresolved error")]):
        lx = [0.2, 2.7, 7.5][k]
        ax.scatter(lx, -2.55, s=150, color=fc, edgecolor=ec, lw=1.1)
        ax.text(lx + 0.3, -2.55, lab, va="center", fontsize=10, color=INK)
    save(fig, "background.png")


# ---------------------------------------------------------------- 2. decomposition
def decomposition():
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(-0.15, 4.2); ax.axis("off")
    fs = 34
    parts = [(r"$\pi^{\mathrm{P}}_t\;=\;$", INK), (r"$\pi^{\mathrm{S}}_t$", NEW), (r"$\;+\;$", INK),
             (r"$(1-\pi^{\mathrm{S}}_t)\,\kappa_t$", "#b0664f")]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widths, texts = [], []
    for s, c in parts:
        tt = ax.text(0, 2.9, s, fontsize=fs, color=c, va="center")
        bb = tt.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        widths.append(bb.width); texts.append(tt)
    x = (10 - sum(widths)) / 2
    spans = []
    for tt, w in zip(texts, widths):
        tt.set_x(x); spans.append((x, x + w)); x += w
    labels = [(1, "whether the error\nis new", NEW), (3, "whether the error\nis inherited", "#b0664f")]
    for i, lab, c in labels:
        a, b = spans[i]
        ax.plot([a + 0.05, a + 0.05, b - 0.05, b - 0.05], [2.42, 2.3, 2.3, 2.42], color=c, lw=1.6)
        mid = (a + b) / 2
        arrow(ax, (mid, 2.28), (mid, 1.72), color=c, lw=1.6, ms=16)
        ax.text(mid, 1.62, lab, ha="center", va="top", fontsize=13, color=c, fontweight="bold", linespacing=1.25)
    ax.add_patch(FancyBboxPatch((1.4, -0.08), 7.2, 0.8, boxstyle="round,pad=0,rounding_size=0.12", fc=RAISE, ec=LINE, lw=1.2))
    ax.text(5, 0.32, r"$\kappa_t=\mathbb{P}_{D^{\mathrm{P}}}\left(y^{\mathrm{P}}_t=1\;\middle|\;y^{\mathrm{S}}_t=0,\ "
            r"\mathbf{Q},\mathbf{S}_{1:t}\right)$", ha="center", va="center", fontsize=17, color=INK)
    save(fig, "decomposition.png")


# ---------------------------------------------------------------- 3a. step read-outs (animated) + AUROC by domain
STEP_AUROC = {  # Qwen3.8-27B Step TEST AUROC; None = not run yet
    "Streaming": [0.8716, 0.8720, 0.8145, 0.7852, 0.8953],
    "SAPLMA": [0.8822, 0.8607, 0.8255, 0.8046, 0.9022],
    "HaMI": [0.8875, 0.8735, None, None, 0.9080],
}
DOMAINS = ["Logic", "Factuality", "Science", "Math", "Multi-hop"]


def step_readouts():
    n = 12
    idx = np.arange(n)
    readouts = [
        ("SAPLMA", "last token", SAPLMA, np.eye(n)[-1]),
        ("Streaming", "fixed weights,\nrising to the end", STREAMING, np.exp(3 * idx / (n - 1)) / np.exp(3 * idx / (n - 1)).sum()),
        ("HaMI", "learned weights on\nselected tokens", HAMI, np.array([.02, .03, .02, .2, .05, .02, .3, .04, .02, .2, .05, .05])),
    ]
    for _, _, _, w in readouts:
        w /= w.max()
    fig = plt.figure(figsize=(12, 4.8), dpi=100)
    axl = fig.add_axes([0.0, 0.0, 0.56, 1.0]); axl.axis("off"); axl.set_xlim(-5.6, n + 2.4); axl.set_ylim(-0.6, 9.6)
    axr = fig.add_axes([0.63, 0.2, 0.36, 0.62])
    axl.text(-5.5, 9.15, "Three read-outs turn the tokens of one step into a step feature", fontsize=12.5, color=INK,
             fontweight="bold")
    rows_y = [6.2, 3.55, 0.9]
    tokens, bars, outs = [], [], []
    for (name, desc, col, w), y in zip(readouts, rows_y):
        axl.text(-5.5, y + 0.45, name, fontsize=12, color=col, fontweight="bold", va="center")
        axl.text(-5.5, y + 0.05, desc, fontsize=9, color=INK2, va="top", linespacing=1.2)
        row_t, row_b = [], []
        for i in range(n):
            p = Rectangle((i - 0.38, y - 0.35), 0.76, 0.6, fc=CLEAN_FILL, ec=INK2, lw=0.8, alpha=0)
            axl.add_patch(p); row_t.append(p)
            b = Rectangle((i - 0.3, y + 0.32), 0.6, 0.0, fc=col, ec="none")
            axl.add_patch(b); row_b.append(b)
        o = FancyBboxPatch((n + 0.4, y - 0.35), 1.5, 0.6, boxstyle="round,pad=0,rounding_size=0.1", fc=col, ec="none", alpha=0)
        axl.add_patch(o)
        ot = axl.text(n + 1.15, y - 0.05, r"$\mathbf{h}_t$", ha="center", va="center", color="white", fontsize=13, alpha=0)
        tokens.append(row_t); bars.append(row_b); outs.append((o, ot))
    axl.text(n / 2 - 0.5, -0.45, "tokens of step $t$", ha="center", fontsize=10, color=INK2)
    # right: AUROC by domain
    x = np.arange(len(DOMAINS)); wbar = 0.26
    for k, (name, col) in enumerate([("Streaming", STREAMING), ("SAPLMA", SAPLMA), ("HaMI", HAMI)]):
        vals = STEP_AUROC[name]
        for i, v in enumerate(vals):
            pos = x[i] + (k - 1) * wbar
            if v is None:
                axr.text(pos, 0.762, "n/a", ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)
                continue
            axr.bar(pos, v - 0.75, wbar - 0.03, bottom=0.75, color=col, edgecolor=SURFACE, lw=0.8, zorder=3,
                    label=name if i == 0 else None)
    for i in range(len(DOMAINS)):
        vals = {k: v[i] for k, v in STEP_AUROC.items() if v[i] is not None}
        best = max(vals, key=vals.get)
        k = ["Streaming", "SAPLMA", "HaMI"].index(best)
        axr.text(x[i] + (k - 1) * wbar, vals[best] + 0.003, "★", ha="center", va="bottom", fontsize=11, color=INK)
    axr.set_xticks(x, DOMAINS, fontsize=9.5, color=INK); axr.set_ylim(0.75, 0.93)
    axr.set_ylabel("Step TEST AUROC (axis from 0.75)", fontsize=9.5, color=INK2)
    axr.tick_params(axis="y", labelsize=9, colors=INK2, length=0); axr.tick_params(axis="x", length=0)
    axr.grid(axis="y", color="#e6dfd0", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        axr.spines[s].set_visible(False)
    axr.spines["bottom"].set_color(INK2)
    axr.legend(ncol=3, frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    axr.set_title("Qwen3.8-27B: the best read-out (★) differs by domain", fontsize=11, color=INK, loc="left", pad=30)

    frames_tok, frames_w, frames_hold = n, 14, 14

    def update(f):
        shown = min(f + 1, n)
        grow = np.clip((f - frames_tok) / (frames_w - 4), 0, 1)
        for r, (row_t, row_b) in enumerate(zip(tokens, bars)):
            w = readouts[r][3]
            for i in range(n):
                row_t[i].set_alpha(1.0 if i < shown else 0.0)
                row_b[i].set_height(1.1 * w[i] * grow)
            outs[r][0].set_alpha(1.0 if grow >= 1 else 0.0)
            outs[r][1].set_alpha(1.0 if grow >= 1 else 0.0)
        return []

    anim = FuncAnimation(fig, update, frames=frames_tok + frames_w + frames_hold, interval=120)
    anim.save(OUT / "step_readouts.gif", writer=PillowWriter(fps=8))
    if len(__import__("sys").argv) > 1:  # optional still frame for checking the layout
        update(frames_tok + frames_w + frames_hold - 1)
        fig.savefig(__import__("sys").argv[1], bbox_inches="tight", pad_inches=0.1, dpi=110)
    plt.close(fig)
    print("wrote", OUT / "step_readouts.gif")


# ---------------------------------------------------------------- 3b. prefix challenge
def prefix_challenge():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.1), dpi=200, gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.35})
    a.barh([0], [15.3], color=NEW, height=0.5, edgecolor=SURFACE, lw=1.5)
    a.barh([0], [84.7], left=[15.3], color=INHERITED, height=0.5, edgecolor=SURFACE, lw=1.5)
    a.text(7.65, 0, "new\n15%", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    a.text(57.6, 0, "inherited  85%", ha="center", va="center", color=INK, fontsize=13, fontweight="bold")
    a.set_xlim(0, 100); a.set_ylim(-0.6, 0.6); a.axis("off")
    a.set_title("Prefix errors: mostly inherited, not new", loc="left", fontsize=11.5, color=INK, pad=8)
    a.text(0, -0.55, "1.45M labelled steps, 3 generators x 5 domains", fontsize=9, color=INK2, va="top")
    labels = ["inherited vs. fresh clean\n(no error so far)", "inherited vs. recovered clean\n(earlier error resolved)"]
    vals = [0.933, 0.661]
    cols = ["#8a94a8", NEW]
    for i, (v, c) in enumerate(zip(vals, cols)):
        b.barh(i, v - 0.5, left=0.5, color=c, height=0.5, zorder=3)
        b.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=12, color=INK, fontweight="bold")
    b.set_yticks([0, 1], labels, fontsize=9.5, color=INK); b.invert_yaxis()
    b.set_xlim(0.5, 1.0); b.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    b.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.set_xlabel("AUROC of existing detectors (mean of 5; 0.5 = chance)", fontsize=9.5, color=INK2)
    b.tick_params(colors=INK2, length=0, labelsize=9); b.grid(axis="x", color="#e6dfd0", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        b.spines[s].set_visible(False)
    b.spines["bottom"].set_color(INK2)
    b.set_title("Detectors cannot tell whether an error is resolved", loc="left", fontsize=11.5, color=INK, pad=8)
    save(fig, "prefix_challenge.png")


# ---------------------------------------------------------------- 3c. chain summary
def chain():
    fig, ax = canvas(11, 2.3, (0, 11), (0, 2.3))
    items = [("Step", r"read-out $\rightarrow\ \pi^{\mathrm{S}}_t$", "important", RAISE, INK2),
             ("Prefix", "$\\kappa_t$: is an earlier error\nstill unresolved?", "core", "#f6e3da", NEW),
             ("Answer", r"$\alpha_1,\alpha_0$ on top of $\pi^{\mathrm{P}}_T$", "tentative", RAISE, LINE)]
    for k, (name, what, tag, fc, ec) in enumerate(items):
        x = 0.2 + k * 3.7
        box(ax, x, 0.35, 3.0, 1.6, fc=fc, ec=ec, lw=1.8 if tag == "core" else 1.2, r=0.15)
        ax.text(x + 0.2, 1.62, name, fontsize=14, color=INK, fontweight="bold", va="center")
        ax.text(x + 2.8, 1.62, tag, fontsize=9.5, color=NEW if tag == "core" else INK2, va="center", ha="right",
                family="DejaVu Sans Mono")
        ax.text(x + 0.2, 0.95, what, fontsize=11, color=INK, va="center", linespacing=1.3)
        if k < 2:
            arrow(ax, (x + 3.05, 1.15), (x + 3.65, 1.15), color=INK2, lw=1.6, ms=16)
    save(fig, "chain.png")


# ---------------------------------------------------------------- 4a. method: learnable aggregation
def step_method():
    fig, ax = canvas(11, 2.7, (0, 11), (0, 2.7))
    rng = np.random.default_rng(3)
    w = np.array([.04, .05, .03, .18, .06, .05, .22, .07, .05, .15, .05, .05])
    for i in range(12):
        ax.add_patch(Rectangle((0.25 + i * 0.27, 0.6), 0.22, 0.45, fc=CLEAN_FILL, ec=INK2, lw=0.7))
        ax.add_patch(Rectangle((0.25 + i * 0.27, 1.12), 0.22, 1.1 * w[i] / w.max(), fc=NEW, ec="none"))
    ax.text(1.85, 0.32, r"token states $\mathbf{h}_{t,i}$", ha="center", fontsize=10.5, color=INK2)
    ax.text(1.85, 2.42, r"weights $a_{t,i}=\mathrm{softmax}_i\,\mathrm{MLP}(\mathbf{h}_{t,i})$", ha="center", fontsize=10.5, color=INK)
    arrow(ax, (3.6, 1.15), (4.4, 1.15))
    box(ax, 4.45, 0.75, 2.6, 0.8, r"$\mathbf{e}_t=\sum_{i}\;a_{t,i}\,\mathbf{h}_{t,i}$", fs=13)
    ax.text(5.75, 0.45, "weighted pooling", ha="center", fontsize=9.5, color=INK2)
    arrow(ax, (7.1, 1.15), (7.8, 1.15))
    box(ax, 7.85, 0.75, 1.3, 0.8, "linear", fs=11)
    arrow(ax, (9.2, 1.15), (9.8, 1.15))
    ax.text(10.35, 1.15, r"$\pi^{\mathrm{S}}_t$", ha="center", va="center", fontsize=18, color=NEW)
    save(fig, "step_method.png")


# ---------------------------------------------------------------- 4b. method: kappa on Step-clean positions
def prefix_method():
    fig, ax = canvas(11, 3.1, (0, 11), (0, 3.1))
    step = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    prefix = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    for t in range(10):
        used = step[t] == 0
        x = 0.35 + t * 0.42
        ax.scatter(x, 2.45, s=170, color=NEW if step[t] else (INHERITED if prefix[t] else "white"),
                   edgecolor=INK2, lw=1, alpha=1 if used else 0.25, zorder=3)
        if not used:
            ax.text(x, 2.45, "×", ha="center", va="center", fontsize=13, color=INK, zorder=4)
            ax.text(x, 2.85, "new", ha="center", fontsize=8, color=INK2)
        if used:
            ax.add_patch(Rectangle((x - 0.17, 1.95), 0.34, 0.2, fc=INHERITED if prefix[t] else CLEAN_FILL, ec="none"))
    ax.text(2.25, 1.6, r"train and test only where $y^{\mathrm{S}}_t=0$", ha="center", fontsize=10.5, color=INK)
    ax.text(2.25, 1.22, "target: inherited vs. clean (incl. recovered)", ha="center", fontsize=9.5, color=INK2)
    arrow(ax, (4.55, 2.2), (5.15, 2.2))
    for k, (name, col) in enumerate([("SAPLMA", SAPLMA), ("Streaming", STREAMING), ("HaMI", HAMI)]):
        box(ax, 5.2, 2.55 - k * 0.55, 1.6, 0.42, name, fc=RAISE, ec=col, fs=10.5, color=col, lw=1.4)
    ax.text(6.0, 0.95, "read-outs for $\\kappa_t$", ha="center", fontsize=9.5, color=INK2)
    arrow(ax, (6.85, 2.2), (7.45, 2.2))
    ax.text(7.95, 2.2, r"$\kappa_t$", ha="center", va="center", fontsize=18, color="#b0664f")
    box(ax, 7.4, 0.35, 3.45, 0.75, r"$\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t+(1-\pi^{\mathrm{S}}_t)\,\kappa_t$", fs=13)
    arrow(ax, (7.95, 1.85), (7.95, 1.15))
    ax.text(9.6, 1.55, r"frozen Step $\pi^{\mathrm{S}}_t$", ha="center", fontsize=10, color=INK2)
    arrow(ax, (9.6, 1.42), (9.6, 1.15))
    save(fig, "prefix_method.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    background(); decomposition(); step_readouts(); prefix_challenge(); chain(); step_method(); prefix_method()
