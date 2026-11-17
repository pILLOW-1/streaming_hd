"""Plot within-Step pooling weight on the four evaluation domains."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


DOMAINS = ("Logic", "Factuality", "Science", "Multi-hop")
STYLES = (
    ("learned", "#00737B", "-", "Learnable aggregation"),
    ("fixed", "#B95724", "--", "Fixed Streaming"),
)


def plot(rows, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#172D36",
                         "axes.labelcolor": "#172D36", "axes.titlecolor": "#172D36"})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.965, bottom=.135, top=.77, hspace=.46, wspace=.22)
    fig.text(.075, .941, "Where pooling puts weight within an erroneous Step",
             fontsize=19, fontweight="bold")
    fig.text(.075, .895, "Qwen Step-only · VAL errors only · Each Step contributes equally",
             fontsize=10.5, color="#44606A")
    fig.legend(handles=[Line2D([], [], color=color, linestyle=ls, linewidth=2.6, label=label)
                        for _, color, ls, label in STYLES], loc="upper left",
               bbox_to_anchor=(.065, .874), ncol=2, frameon=False, fontsize=10)
    peak = max(max(r["mean_weight_mass"][arm]) for r in rows.values() for arm, *_ in STYLES)
    top = (np.ceil(peak * 10) / 10) + .01
    for domain, ax in zip(DOMAINS, axes.flat):
        row = rows[domain]
        edges = row["bin_edges"]
        ax.axvspan(.8, 1, facecolor="#E9EFF1", zorder=0)
        ax.axhline(.1, color="#A5B5BA", linewidth=.95, zorder=1)
        ax.grid(axis="y", linewidth=.65, color="#DDE6E7")
        for arm, color, ls, _ in STYLES:
            ax.stairs(row["mean_weight_mass"][arm], edges, baseline=None, color=color, linestyle=ls,
                      linewidth=2.6, zorder=4)
        ax.set_title(domain, loc="left", fontweight="bold", fontsize=11.5, pad=8)
        ax.text(1, 1.025, f'{row["positions"]:,} Steps', transform=ax.transAxes,
                va="bottom", ha="right", fontsize=9, color="#44606A")
        ax.set(xlim=(0, 1), ylim=(0, top), xticks=[0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(labelsize=9, length=0)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Relative position in Step   ·   0% start     100% end", fontsize=9.5)
    fig.supylabel("Mean share of total Step weight per 10% bin", x=.025, y=.45, fontsize=10)
    fig.text(.075, .034, "The shaded region marks the final 20%. A flat distribution gives 10% weight to every bin.",
             fontsize=9, color="#44606A")
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot({d: json.loads((args.result_dir / f"{d}.json").read_text()) for d in DOMAINS}, args.output)
