"""Uniform Step call boundary without changing the exported model equations."""
import torch
from . import models

STEP_MODELS = {
    'learnable_aggregation': models.AggregationOnlyStreaming,
    'learnable_aggregation_representation': models.SharedRepresentationStreaming,
    'learned_token': models.LearnedTokenStep,
    'self_attention': models.StandardSelfAttentionStep,
    'token_relation_residual': models.TokenRelationResidualStep,
    'content_relation': models.ContentRelationResidualStep,
}
PREFIX_MODELS = {
    'scalar_gate': models.DynamicPrefixGate,
    'vector_gate': models.FullVectorPrefixGate,
    'content_relation_gate': models.RelationConditionedPrefixGate,
    'direct_mlp': lambda width: models.DirectMLPPrefix(width, 1e-12),
}


def step_forward(model, bags, device):
    """Return logits and the exact representation passed to the final probe."""
    lengths=[len(x) for x in bags]
    if not bags or min(lengths)<=0:raise ValueError('Nonempty Step token bags required')
    tokens=torch.cat(bags);offsets=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)))
    indices=torch.arange(len(bags));captured=[]
    hook=model.probe.register_forward_pre_hook(lambda module,args: captured.append(args[0]))
    try:
        if hasattr(model,'forward_bags'):
            logits=model.forward_bags(tokens,offsets,indices,device)[0]
        elif hasattr(model,'forward_cache'):
            logits=model.forward_cache({'tokens':tokens,'offsets':offsets},indices,device,relation_microbatch_size=8)[0]
        elif hasattr(model,'pool_indices'):
            _,logits,_=model.pool_indices({'tokens':tokens,'offsets':offsets},indices,device)
        else:
            states=torch.nn.utils.rnn.pad_sequence([b.to(device).float() for b in bags],batch_first=True)
            mask=torch.arange(states.shape[1],device=device)[None,:]<torch.tensor(lengths,device=device)[:,None]
            logits=model.forward_padded(states,mask)[0]
    finally:
        hook.remove()
    representation=torch.cat(captured)
    if len(logits)!=len(bags) or len(representation)!=len(bags):raise ValueError('Step output alignment failure')
    return logits,representation
