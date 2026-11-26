#!/usr/bin/env python3
"""How large the label problem is in each model-domain combination (TEST label check).
Bar length = share of reviewed supervised positions (Step/Prefix label pairs) with a proposed correction;
segments split that share by error type, typed by (original pair -> proposed pair).
Input: the TEST label-check distribution JSON (private, on Cetus). Outputs: weekly/assets/2026-11-27/{trajectory,position}_distribution.png"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           '/shared/homes/u26487468/agent_uncertainty_quantification/experiments/streaming_hd/runs/audit/jialong/TEST_MODEL_DOMAIN_PROBLEM_DISTRIBUTION.json')
OUT = Path(__file__).resolve().parents[1] / 'assets/2026-11-27'
TYPES = [  # (key in source, legend label), fixed order = fixed colour
    ('疑似假恢复／漏掉仍有效旧错误', 'False recovery (0,0)→(0,1)'),
    ('继承或重复旧错误误标为新错误', 'Repetition as new error (1,1)→(0,1)'),
    ('Prefix疑似误报／实际已无有效错误', 'Spurious Prefix (0,1)→(0,0)'),
    ('Step与Prefix均疑似误报', 'Spurious error (1,1)→(0,0)'),
    ('漏标新错误（同时漏标Prefix）', 'Missed error (0,0)→(1,1)'),
    ('已有错误期间漏标额外新错误', 'Missed error in active error (0,1)→(1,1)'),
]
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#8172B3', '#937860', '#DA8BC3']  # muted academic palette
MODELS = [('deepseek_r1_distill_llama_8b', 'DS-L8'), ('gemma_4_31b_it', 'Gemma-4-31B-it'), ('qwen3_8_27b', 'Qwen3.8-27B')]
DOMAINS = [('Math', 'Math'), ('Science', 'Science'), ('Logic', 'Logic'), ('Multi-hop', 'Multi-hop'), ('Factuality', 'Fact')]

d = json.load(open(SRC)); rows = {(r['model'], r['domain']): r for r in d['rows']}
CLEAN, BAD = '#E6E6E6', '#C44E52'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9})
WP = dict(edgecolor='white', linewidth=1.0)

def grid(draw, legend, title, fname):
    fig, axes = plt.subplots(3, 5, figsize=(11, 7.4))
    for i, (m, mn) in enumerate(MODELS):
        for j, (dm, dn) in enumerate(DOMAINS):
            ax = axes[i, j]; r = rows[(m, dm)]
            note = draw(ax, r)
            t = dn + ('' if r['reviewed'] == r['total'] else f"$^*$")
            ax.set_title(t, fontsize=10)
            ax.text(0, -1.32, note, ha='center', fontsize=8.5, color='#333')
            if j == 0: ax.text(-1.85, 0, mn, rotation=90, ha='center', va='center', fontsize=10.5)
    fig.legend(*legend, loc='lower center', ncol=len(legend[0]) if len(legend[0]) <= 3 else 3, frameon=False, fontsize=9, title=title, title_fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.14 if len(legend[0]) > 3 else 0.09, hspace=0.42, wspace=0.05)
    fig.savefig(OUT / fname, dpi=170, facecolor='white'); print(OUT / fname)

def traj(ax, r):
    bad = r['affected'] / r['reviewed']
    ax.pie([bad, 1 - bad], colors=[BAD, CLEAN], startangle=90, counterclock=False, wedgeprops=WP)
    return f"{100 * bad:.0f}% of {r['reviewed']} traj."

def pos(ax, r):
    v = [r['types'].get(k, {}).get('positions', 0) for k, _ in TYPES]
    ax.pie(v + [r['positions'] - sum(v)], colors=COLORS + [CLEAN], startangle=90, counterclock=False, wedgeprops=WP)
    return f"{100 * sum(v) / r['positions']:.0f}% of {r['positions']:,} positions"

P = plt.matplotlib.patches.Patch
OUT.mkdir(parents=True, exist_ok=True)
grid(traj, ([P(color=BAD), P(color=CLEAN)], ['With a proposed correction', 'No correction']), None, 'trajectory_distribution.png')
grid(pos, ([P(color=c) for c in COLORS + [CLEAN]], [l for _, l in TYPES] + ['No correction']),
     'Type: original (Step, Prefix) → proposed (Step, Prefix)', 'position_distribution.png')
