from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
domains=('Logic','Factuality','Science','Multi-hop')
colors={'learned':'#087e8b','fixed':'#d76b2d'}
labels={'learned':'Learnable aggregation','fixed':'Fixed Streaming'}
results={d: json.loads((root/'result'/f'{d}.json').read_text()) for d in domains}
for figure in ('effective_by_length','argmax_position'):
    fig, axes=plt.subplots(2,2,figsize=(9,6.3),layout='constrained',sharey=True)
    for domain,ax in zip(domains,axes.flat):
        groups=results[domain]['summary']
        for arm,color in colors.items():
            if figure == 'effective_by_length':
                bins=[(k,v) for k,v in groups.items() if k!='all' and v['positions']]
                x=np.arange(len(bins))
                q=np.array([v['metrics']['effective_ratio'][arm] for k,v in bins])
                ax.errorbar(x+(-.08 if arm=='learned' else .08),q[:,1],
                            yerr=[q[:,1]-q[:,0],q[:,2]-q[:,1]],fmt='o',capsize=3,color=color,label=labels[arm])
                ax.set_xticks(x,[k for k,v in bins],rotation=20)
                ax.set(xlabel='Step length (tokens)',ylim=(0,1.03))
            else:
                counts=np.array(groups['all']['argmax_deciles'][arm])
                ax.stairs(counts/counts.sum(),np.linspace(0,1,11),label=labels[arm],color=color,linewidth=2)
                ax.set(xlabel='Relative position of maximum weight',ylim=(0,1.03))
        ax.set_title(domain)
        ax.grid(axis='y',alpha=.22)
    axes[0,0].set_ylabel('Effective token fraction' if figure=='effective_by_length' else 'Fraction of Steps')
    axes[1,0].set_ylabel(axes[0,0].get_ylabel())
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(root/f'{figure}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
