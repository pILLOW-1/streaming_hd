"""VAL ranking with explicit unresolved-tie detection."""
def best_snapshot(rank, state, best):
    """No preference for smaller layers or earlier updates."""
    import torch
    if best is None or rank>best['rank']:
        return dict(rank=rank,state={k:v.detach().cpu().clone() for k,v in state.items()},tie=False)
    if rank==best['rank'] and any(not torch.equal(v.detach().cpu(),best['state'][k]) for k,v in state.items()):
        best['tie']=True
    return best

