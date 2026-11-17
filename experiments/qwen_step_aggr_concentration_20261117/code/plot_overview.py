"""Plot paired four-domain VAL summaries from the published aggregate JSON."""
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


def overview(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.1))
    fig.subplots_adjust(left=.15, right=.98, bottom=.12, top=.77, hspace=.66, wspace=.40)
    fig.text(.06, .953, 'How does learned aggregation distribute token weight?', fontsize=18,
             fontweight='bold', color='#102A33')
    fig.text(.06, .909, 'Qwen Step-only · Four domains · Same VAL steps for both methods',
             fontsize=10.5, color='#465960')
    handles = [Line2D([], [], color=color, marker=marker, linestyle='none', label=label, markersize=7)
               for color, marker, label in ((PALETTE['learned'], 'o', 'Learnable aggregation'),
                                             (PALETTE['fixed'], 'D', 'Fixed Streaming'))]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.051, .89), ncol=2, frameon=False)
    for ax, (key, title, subtitle) in zip(axes.flat, METRICS):
        ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', pad=19)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8.2, color='#516469')
        y = np.arange(len(DOMAINS))
        for arm, offset, marker in (('learned', -.17, 'o'), ('fixed', .17, 'D')):
            q = np.array([data[d]['summary']['all']['metrics'][key][arm] for d in DOMAINS])
            ax.errorbar(q[:, 1], y+offset, xerr=[q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]], fmt=marker,
                        color=PALETTE[arm], markersize=5.4, linewidth=1.45, capsize=2.6)
        ax.set(yticks=y, yticklabels=DOMAINS, xlim=(0, 1.01), ylim=(3.55, -.55),
               xticks=[0, .25, .50, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(labelsize=9, length=0)
        ax.grid(axis='x', color='#DCE5E8', linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.06, .044, 'Markers are VAL-step medians. Bars show the middle 50% of steps (interquartile range).',
             fontsize=9, color='#465960')
    fig.savefig(out, dpi=190, facecolor='white')
    plt.close(fig)


def paired_by_length(data, out):
    bins = ('1', '2–4', '5–16', '17–64', '65–256', '257+')
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.7))
    fig.subplots_adjust(left=.128, right=.865, bottom=.13, top=.80, hspace=.53, wspace=.36)
    fig.text(.065, .950, 'Where does learned aggregation differ from fixed Streaming?',
             fontsize=17, fontweight='bold', color='#102A33')
    fig.text(.065, .899, 'Median paired difference within each Step length group · percentage points',
             fontsize=10.5, color='#465960')
    for ax, (key, title, _) in zip(axes.flat, METRICS):
        values = np.full((4,6), np.nan)
        for i, domain in enumerate(DOMAINS):
            groups = list(data[domain]['summary'].items())
            groups = [(name,g) for name,g in groups if name != 'all']
            assert len(groups) == len(bins)
            for j, (_, group) in enumerate(groups):
                q = group['metrics'][key]['paired_delta']
                if q is not None:
                    values[i,j] = 100*q[1]
        palette = plt.get_cmap('RdBu').copy()
        palette.set_bad('#F0F3F4')
        mesh = ax.imshow(values, cmap=palette, vmin=-35, vmax=35, aspect='auto')
        for (i,j), value in np.ndenumerate(values):
            label = f'{value:+.1f}' if np.isfinite(value) and abs(value)>=.05 else (
                '0.0' if np.isfinite(value) else '–')
            text_color = 'white' if np.isfinite(value) and abs(value)>=24 else '#102A33'
            ax.text(j,i,label,ha='center',va='center',color=text_color,fontsize=8.5)
        ax.set(xticks=range(len(bins)),xticklabels=bins,yticks=range(len(DOMAINS)),yticklabels=DOMAINS)
        ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
        ax.tick_params(labelsize=8,length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(mesh, cax=fig.add_axes([.899,.20,.016,.55]), extend='both')
    bar.ax.set_ylabel('Learnable − fixed (pp)',fontsize=9)
    fig.text(.065,.047,'Each panel groups the same VAL steps by token count. A dash marks an empty group.',
             fontsize=9,color='#465960')
    fig.savefig(out,dpi=190,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='Directory of result JSON files or a single domain-keyed JSON')
    parser.add_argument('output', type=Path)
    parser.add_argument('--paired-output', type=Path)
    args = parser.parse_args()
    data = (json.loads(args.input.read_text()) if args.input.is_file()
            else {d: json.loads((args.input/f'{d}.json').read_text()) for d in DOMAINS})
    overview(data, args.output)
    if args.paired_output:
        paired_by_length(data, args.paired_output)
