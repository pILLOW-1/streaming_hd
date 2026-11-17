"""Plot paired four-domain VAL summaries on ground-truth error Steps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


DOMAINS = ('Logic', 'Factuality', 'Science', 'Multi-hop')
METRICS = (
    ('effective_ratio', 'Effective token fraction', 'Lower means more concentrated'),
    ('top1_mass', 'Top-1 token weight', 'Largest single token weight'),
    ('top3_mass', 'Top-3 token weight', 'Combined weight of up to three tokens'),
    ('last20_mass', 'Final 20% of tokens', 'Weight assigned to the end of the step'),
)
PALETTE = {'learned': '#006D77', 'fixed': '#CB6D31'}


def overview(data, out, split):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.15, right=.98, bottom=.12, top=.77, hspace=.66, wspace=.40)
    fig.text(.06, .953, f'{split.upper()} · Token-weight concentration on error Steps', fontsize=18,
             fontweight='bold', color='#102A33')
    fig.text(.06, .909, 'Qwen Step-only · Ground-truth error Steps only',
             fontsize=10.5, color='#465960')
    handles = [Line2D([], [], color=color, marker=marker, linestyle='none', label=label, markersize=7)
               for color, marker, label in ((PALETTE['learned'], 'o', 'Learnable aggregation'),
                                             (PALETTE['fixed'], 'D', 'Fixed Streaming'))]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.051, .89), ncol=2, frameon=False)
    for ax, (key, title, subtitle) in zip(axes.flat, METRICS):
        ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', pad=19)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8.2, color='#516469')
        y = np.arange(len(DOMAINS))
        labels = [f'{d} ({data[d]["positions"]:,})' for d in DOMAINS]
        for arm, offset, marker in (('learned', -.17, 'o'), ('fixed', .17, 'D')):
            q = np.array([data[d]['summary']['all']['metrics'][key][arm] for d in DOMAINS])
            ax.errorbar(q[:, 1], y+offset, xerr=[q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]], fmt=marker,
                        color=PALETTE[arm], markersize=5.4, linewidth=1.45, capsize=2.6)
        ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.01), ylim=(3.55, -.55),
               xticks=[0, .25, .50, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(labelsize=9, length=0)
        ax.grid(axis='x', color='#DCE5E8', linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.06, .044, 'Markers are medians. Bars show the middle 50% of error Steps. Counts are in parentheses.',
             fontsize=9, color='#465960')
    fig.savefig(out, dpi=190, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('train_output', type=Path)
    parser.add_argument('val_output', type=Path)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    for split, out in [('train', args.train_output), ('val', args.val_output)]:
        overview(rows[split], out, split)
