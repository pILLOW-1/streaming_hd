"""Plot the position of each method's highest scoring token in error Steps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def plot(data, output):
    fig, axes = plt.subplots(len(data), 1, figsize=(10.2, 7.2))
    fig.subplots_adjust(left=.23, right=.96, bottom=.12, top=.82, hspace=.67)
    fig.text(.075, .95, "Where the highest scoring token falls within an error Step",
             fontsize=17, color="#172D36", weight="bold")
    fig.text(.075, .905, "Qwen · VAL errors only · Each row sums to 100%",
             fontsize=10.5, color="#44606A")
    colors = matplotlib.colormaps["Blues"]
    norm = Normalize(0, .50, clip=True)
    for (domain, row), ax in zip(data.items(), axes):
        values = np.asarray([row["hami"], row["learned"], row["fixed"]])
        ax.imshow(values, cmap=colors, norm=norm, aspect="auto", interpolation="nearest")
        for (i, j), v in np.ndenumerate(values):
            value = f"{v:.1%}".replace(".0%", "%") if v else "·"
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if v >= .31 else "#172D36", fontsize=10)
        ax.set_title(f'{domain}  ·  {row["positions"]:,} Steps', loc="left",
                     color="#172D36", weight="bold")
        ax.set_yticks([0, 1, 2], ["HaMI", "Learnable aggregation", "Fixed Streaming"])
        ax.set_xticks(np.arange(10), [f"{i}–{i+10}%" for i in range(0, 100, 10)])
        ax.tick_params(length=0, labelcolor="#172D36")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Relative position in the Step, from start to end", labelpad=10)
    fig.text(.075, .034, "Color saturates at 50%. Values are percentages of error Steps.",
             fontsize=9, color="#44606A")
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.figure)
