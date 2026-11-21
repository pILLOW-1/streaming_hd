#!/usr/bin/env python3
"""Figures and tables for the 2026-11-21 weekly report.

Reads the local diagnostic outputs of the random 6:2:2 split (Streaming TEST predictions) and writes
  weekly/assets/2026-11-21/history_split.png   A_pre vs A_post for Step and Prefix in all 15 combinations
  weekly/assets/2026-11-21/score_dynamics.png  mean score percentile around the first error / first recovery
and prints the Markdown tables used in the report.

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

PRE, POST = "#3f74b8", "#a8402f"          # before any error / after an error (validated pair)
INK, MUTED, GRID = "#2b2b2b", "#8a8a85", "#ececea"
MODELS = {"deepseek_r1_distill_llama_8b": "DS-L8", "gemma_4_31b_it": "Gemma-4-31B-it", "qwen3_8_27b": "Qwen3.8-27B"}
DOMAINS = {"Math": "Math", "Science": "Science", "Logic": "Logic", "Multi-hop": "Multi-hop", "Factuality": "Fact"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#c8c8c4",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})

rows = [r for r in json.loads((REPORTS / "history_split_all.json").read_text()) if r["method"] == "Streaming"]
rows.sort(key=lambda r: (list(MODELS).index(r["model"]), list(DOMAINS).index(r["domain"])))
assert len(rows) == 15


# ---------------------------------------------------------------- Figure 1: dumbbells
fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2), sharey=True)
ypos, labels, y = [], [], 0
for i, r in enumerate(rows):
    if i and r["model"] != rows[i - 1]["model"]:
        y += 0.8                                    # gap between model groups
    ypos.append(y); labels.append(DOMAINS[r["domain"]]); y += 1
ypos = np.array(ypos)
for ax, level, title in zip(axes, ("step", "prefix"), ("Step", "Prefix")):
    pre = np.array([r[level]["A_pre"] for r in rows]); post = np.array([r[level]["A_post"] for r in rows])
    for yy, a, b in zip(ypos, pre, post):
        ax.plot([a, b], [yy, yy], color="#cfcfca", lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(pre, ypos, s=48, color=PRE, zorder=3, edgecolor="white", linewidth=1.2, label="Negatives before any error")
    ax.scatter(post, ypos, s=48, color=POST, zorder=3, edgecolor="white", linewidth=1.2, label="Negatives after an error")
    ax.set_xlim(0.5, 1.0); ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("TEST AUROC against the negative group")
axes[0].set_yticks(ypos, labels); axes[0].invert_yaxis()
starts = [0, 5, 10]
for s in starts:                                   # model names above each group
    axes[0].text(-0.02, ypos[s] - 0.75, MODELS[rows[s]["model"]], transform=axes[0].get_yaxis_transform(),
                 ha="right", va="center", fontsize=10.5, fontweight="bold", color=INK)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(rect=(0.04, 0, 1, 0.94))
fig.savefig(OUT / "history_split.png", dpi=170)
plt.close(fig)


# ---------------------------------------------------------------- Figure 2: score dynamics
dyn = json.loads((REPORTS / "score_dynamics.json").read_text())
K = np.array(dyn["offsets"])
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), sharex=True, sharey=True)
for r, level in enumerate(("Step", "Prefix")):
    for c, event in enumerate(("error", "recovery")):
        ax = axes[r, c]; d = dyn[level]
        for v, color, name in ((1, POST, f"{level} = 1"), (0, PRE, f"{level} = 0")):
            m = np.array([np.nan if x is None else x for x in d[f"{event}_{v}"]["mean"]], float)
            m[np.array(d[f"{event}_{v}"]["cells"]) < 3] = np.nan
            ax.plot(K, m, color=color, lw=2, marker="o", ms=3.5, label=name)
        ax.axhline(d["fresh"], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Negatives before any error")
        ax.axvline(0, color="#d8d8d4", lw=1, zorder=0)
        ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.8); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(-5, 16, 5))
        if r == 0:
            ax.set_title("Aligned at the first error" if event == "error" else "Aligned at the first recovery",
                         color=INK, fontsize=11.5, loc="left")
        if c == 0:
            ax.set_ylabel(f"{level} score percentile")
        if r == 1:
            ax.set_xlabel("Steps relative to the event")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "score_dynamics.png", dpi=170)
plt.close(fig)


# ---------------------------------------------------------------- Markdown tables
def f(x):
    return f"{x:.4f}"


print("| Model | Domain | Step $`A_{\\mathrm{pre}}`$ | Step $`A_{\\mathrm{post}}`$ | Prefix $`A_{\\mathrm{pre}}`$ | Prefix $`A_{\\mathrm{post}}`$ |")
print("|---|---|---:|---:|---:|---:|")
for i, r in enumerate(rows):
    m = MODELS[r["model"]] if i % 5 == 0 else ""
    print(f"| {m} | {DOMAINS[r['domain']]} | {f(r['step']['A_pre'])} | {f(r['step']['A_post'])} | "
          f"{f(r['prefix']['A_pre'])} | {f(r['prefix']['A_post'])} |")


def summary(groups, header):
    print(f"\n| {header} | Step $`A_{{\\mathrm{{pre}}}}`$ | Step $`A_{{\\mathrm{{post}}}}`$ | Prefix $`A_{{\\mathrm{{pre}}}}`$ | Prefix $`A_{{\\mathrm{{post}}}}`$ |")
    print("|---|---:|---:|---:|---:|")
    for name, members in groups:
        vals = [np.mean([r[lv][k] for r in members]) for lv in ("step", "prefix") for k in ("A_pre", "A_post")]
        print(f"| {name} | " + " | ".join(f(v) for v in vals) + " |")


summary([(MODELS[m], [r for r in rows if r["model"] == m]) for m in MODELS], "Model")
summary([(DOMAINS[d], [r for r in rows if r["domain"] == d]) for d in DOMAINS], "Domain")
summary([("All 15 combinations", rows)], "Scope")
