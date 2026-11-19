"""Equivalent batched A2 execution, keeping the original worker as the contract.

Use only for newly dispatched fits. Never replace an in-flight model. Record the
execution implementation separately from the frozen scientific binding.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import worker as original

OriginalBank = original.TokenBank
OriginalHistorical = original.Model.historical
OriginalForward = original.Model.forward_positions
OriginalPredict = original.predict
ARM = None


def metadata(bank):
    if hasattr(bank, '_layout'):
        return bank._layout
    starts = np.empty(len(bank.data.starts), np.int64)
    lengths = starts.copy()
    for offset, origin, spans, hist, _, _ in bank.data.history:
        starts[hist] = [offset+a-origin for a,b in spans]
        lengths[hist] = [b-a for a,b in spans]
    bank._layout = starts, lengths
    return bank._layout


def raw_positions(bank, step_ids):
    starts, lengths = metadata(bank)
    n = lengths[step_ids]
    offsets = np.cumsum(n)-n
    flat = np.repeat(starts[step_ids], n) + np.arange(n.sum()) - np.repeat(offsets, n)
    return flat, n


def gather(bank, indices, device):
    bits = np.ascontiguousarray(bank.raw[indices])
    return torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(device, dtype=torch.float32)


def top_grid(logits, lengths):
    """Length buckets preserve token order and deterministic stable-sort ties."""
    lengths = np.asarray(lengths, dtype=np.int64)
    counts = (lengths*.1).astype(np.int64)+1
    maximum = int(counts.max())
    result = torch.zeros((len(lengths), maximum), dtype=torch.long, device=logits.device)
    offsets = np.cumsum(lengths)-lengths
    widths = np.asarray([1 << (int(n)-1).bit_length() for n in lengths])
    for width in sorted(set(widths)):
        members = np.flatnonzero(widths == width)
        for group in np.array_split(members, max(1, math.ceil(len(members)*int(width)/65536))):
            ids = torch.as_tensor(group, device=logits.device)
            cols = torch.arange(int(width), device=logits.device)
            offsets_t = torch.as_tensor(offsets[group], device=logits.device)
            positions = offsets_t[:,None] + cols[None,:]
            valid = cols[None,:] < torch.as_tensor(lengths[group], device=logits.device)[:,None]
            values = logits[positions.clamp_max(len(logits)-1)].masked_fill(~valid, -torch.inf)
            chosen = torch.argsort(values, dim=1, descending=True, stable=True)
            k = int(counts[group].max())
            result[ids,:k] = offsets_t[:,None] + chosen[:,:k]
    count_tensor = torch.as_tensor(counts, device=logits.device)
    mask = torch.arange(maximum, device=logits.device)[None,:] < count_tensor[:,None]
    return result, mask, count_tensor


def historical(net, bank, tids, needed=None):
    device = net.prefix.weight.device
    ranges = []
    step_ids = []
    cursor = 0
    for tid in tids:
        hist = bank.data.history[tid][3]
        # Only steps strictly before the latest requested position are needed.
        # Unsupervised original steps in that prefix remain included.
        n = hist.stop-hist.start if needed is None else max(needed[tid])
        ranges.append((tid,cursor,cursor+n,n))
        step_ids.extend(range(hist.start,hist.start+n));cursor += n
    previous = net.training
    net.eval()
    memories = {}; probabilities = {}
    try:
        with torch.no_grad():
            if step_ids:
                positions, lengths = raw_positions(bank, np.asarray(step_ids))
                logits = torch.cat([net.scorer(gather(bank,p,device)).flatten()
                                    for p in np.array_split(positions,max(1,math.ceil(len(positions)/8192)))])
                grid, mask, counts = top_grid(logits,lengths)
                selected = logits[grid]
                q = selected.sigmoid().masked_fill(~mask,0).sum(1)/counts
                selected_cpu = grid.cpu().numpy(); mask_cpu = mask.cpu().numpy()
            else:
                q = torch.empty(0,device=device)
            for tid,lo,hi,n in ranges:
                if n == 0:
                    memories[tid] = torch.zeros((1,net.memory.in_features),device=device)
                    probabilities[tid] = q[lo:hi]
                    continue
                if net.arm == 'H1':
                    ids = np.asarray(step_ids[lo:hi])
                    content = torch.from_numpy(np.ascontiguousarray(bank.fixed_e[ids])).to(device)
                else:
                    chosen = selected_cpu[lo:hi][mask_cpu[lo:hi]]
                    # One coalesced transfer and segmented pool per trajectory,
                    # rather than a separate transfer/kernel for each Step.
                    vectors = gather(bank,positions[chosen],device)
                    content = torch.segment_reduce(vectors,'mean',lengths=counts[lo:hi])
                weighted = q[lo:hi,None]*content
                cumulative = weighted.cumsum(0,dtype=torch.float64)
                if needed is None:
                    g = F.pad(cumulative[:-1],(0,0,1,0)).float()
                else:
                    g = F.pad(cumulative,(0,0,1,0)).float()
                memories[tid] = g
                probabilities[tid] = q[lo:hi]
    finally:
        net.train(previous)
    return memories, probabilities


def fast_forward(self, bank, ids):
    device = self.prefix.weight.device
    ids = np.asarray(ids,dtype=np.int64)
    tids = sorted(set(int(bank.tid[i]) for i in ids))
    needed = {tid:[int(bank.local[i]) for i in ids if bank.tid[i] == tid] for tid in tids}
    memories,_ = historical(self,bank,tids,needed)
    global_steps = np.asarray([bank.data.history[int(bank.tid[i])][3].start+int(bank.local[i]) for i in ids])
    positions,lengths = raw_positions(bank,global_steps)
    # The entire supervised token batch passes BN/Dropout together, exactly as
    # the original worker. Never chunk this training-mode scorer invocation.
    logits = self.scorer(gather(bank,positions,device)).flatten()
    grid,mask,counts = top_grid(logits,lengths)
    values = logits[grid]
    logq = torch.logsumexp(F.logsigmoid(values).masked_fill(~mask,-torch.inf),1)-counts.log()
    log1q = torch.logsumexp(F.logsigmoid(-values).masked_fill(~mask,-torch.inf),1)-counts.log()
    g = torch.stack([memories[int(bank.tid[i])][int(bank.local[i])] for i in ids])
    c = bank.prefix[torch.as_tensor(ids,device=device)]
    return (self.prefix(c)+self.memory(g)).flatten(),logq,log1q


def fast_predict(net, bank):
    net.eval()
    q = np.empty(len(bank.data.y),np.float32);k=q.copy()
    with torch.no_grad():
        for start in range(0,len(bank.data.history),16):
            tids=list(range(start,min(start+16,len(bank.data.history))))
            memories,probabilities=historical(net,bank,tids)
            for tid in tids:
                _,_,_,_,slots,choices=bank.data.history[tid]
                idx=torch.as_tensor(choices,device=net.prefix.weight.device)
                z=(net.prefix(bank.prefix[slots])+net.memory(memories[tid][idx])).flatten()
                q[slots]=probabilities[tid][idx].cpu().numpy();k[slots]=z.sigmoid().cpu().numpy()
    return q,k,q+(1-q)*k


class FastBank(OriginalBank):
    def __init__(self,data,layer,device,cached=True):
        super().__init__(data,layer,device,cached)
        if ARM == 'H1':
            import feature_cache as cache
            self.fixed_e=cache.get(cache.identity(data,layer,'step_exp_all_history'),
                lambda:original.w.raw_pools(data,layer,False)[0])


def install():
    original.Model.forward_positions=fast_forward
    original.TokenBank=FastBank
    original.predict=fast_predict


def main():
    global ARM
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('fit',));p.add_argument('domain',choices=('Factuality',));p.add_argument('arm',choices=('H1','H2'));p.add_argument('layer',type=int)
    a=p.parse_args();ARM=a.arm
    gate=Path(__file__).with_name('FAST_EQUIVALENCE_GATE.json')
    pin=json.loads(gate.read_text())
    assert pin['status']=='PASS' and pin['fast_sha256']==original.w.sha(Path(__file__))
    out=original.w.BASE/original.NAME/a.domain/a.arm;out.mkdir(parents=True,exist_ok=True)
    execution=dict(scientific_contract_sha256=original.w.sha(Path(original.__file__)),
                   implementation_sha256=original.w.sha(Path(__file__)),gate=pin,
                   inherited_original_optimizer_and_checkpoint_protocol=True)
    original.w.save_json(out/f'EXECUTION_{a.layer:02d}.json',execution)
    install();original.fit(a.domain,a.arm,a.layer)


if __name__=='__main__':main()
