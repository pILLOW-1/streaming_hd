"""Map native attention blocks to their actual input/output hidden-state indices."""
import numpy as np
import torch
from .attention import ICRScore, js_divergence


def prefix_vector(hidden_states, attentions, reasoning_start, end, *, core_positions, top_p=.1,
                  pooling='mean', use_induction_head=True):
    """attentions maps zero-based decoder indices to native [heads,T,T] weights.

For a hybrid backbone, pass only its true attention blocks. A block l always
uses hidden_states[l] and hidden_states[l+1]; skipped linear-attention blocks
must never renumber the hidden states. No reconstructed attention is accepted.
"""
    if not 0<=reasoning_start<end or not attentions:
        raise ValueError('Nonempty causal reasoning prefix and native attention blocks required')
    a,b,c=(core_positions[k] for k in ('user_prompt_start','user_prompt_end','response_start'))
    if not 0<=a<b<=c<=reasoning_start:
        raise ValueError('Exact native user-prompt/response boundaries required')
    mask=torch.zeros((end,end),dtype=torch.bool)
    mask[a:b,a:b]=True;mask[c:,c:]=True
    k=int(top_p*end)
    if k<2:raise ValueError('Official ICR sample-standard-deviation requires at least two selected tokens')
    vector=[]
    for block,a in sorted(attentions.items()):
        if a.ndim!=3 or a.shape[-2]<end or a.shape[-1]<end or block+1>=len(hidden_states):
            raise ValueError('Native attention/hidden-state layer or token alignment failure')
        attention=a[:,:end,:end].float().clone()
        attention.masked_fill_(~mask.to(a.device),0)
        selected=torch.ones(len(attention),dtype=torch.bool,device=a.device)
        if use_induction_head:
            moments=[ICRScore._calculate_skewness_entropy(None,head) for head in attention]
            skew=torch.tensor([x[0] for x in moments],device=a.device)
            entropy=torch.tensor([x[1] for x in moments],device=a.device)
            selected=(skew>=3)&(entropy<=3)
            minimum=len(attention)//8
            if selected.sum()<minimum:
                selected[:]=False;selected[skew.topk(minimum).indices]=True
        if not selected.any():raise ValueError('Official ICR has no eligible induction head')
        heads=attention[selected,reasoning_start:end]
        if pooling=='mean':pooled=heads.mean(0)
        elif pooling=='max':pooled=heads.max(0).values
        elif pooling=='min':pooled=heads.min(0).values
        else:raise ValueError('Unknown ICR head pooling')
        before=hidden_states[block][:end].float();after=hidden_states[block+1][:end].float()
        scores=[]
        for j,weights in enumerate(pooled,reasoning_start):
            values,indices=weights.topk(k)
            keys=before[indices]
            contribution=((after[j]-before[j])*keys).sum(-1)/(torch.linalg.vector_norm(keys,dim=-1)+1e-8)
            scores.append(js_divergence(contribution,values))
        vector.append(float(np.mean(scores)))
    result=np.asarray(vector,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError('Undefined native ICR score')
    return result
