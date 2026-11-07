"""Lazy, hash-verified retained-Step token access; reuse native extraction outputs."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path


class TokenCache:
    def __init__(self, root, records, layer, max_cached_trajectories=2):
        self.root=Path(root);self.records=records;self.layer=layer
        self.cache=OrderedDict();self.max_cached=max_cached_trajectories
        self.positions=[(i,j) for i,r in enumerate(records) for j in range(len(r['retained_original_step_ids']))]
        self.trajectory_slices=[];offset=0
        for row in records:
            n=len(row['retained_original_step_ids'])
            self.trajectory_slices.append(slice(offset,offset+n));offset+=n

    def __len__(self):
        return len(self.positions)

    def _trajectory(self, i):
        import torch
        if i in self.cache:
            self.cache.move_to_end(i);return self.cache[i]
        row=self.records[i];folder=self.root/row['sample_id']
        manifest=json.loads((folder/'MANIFEST.json').read_text())
        if manifest['retained_original_step_ids'] != row['retained_original_step_ids']:
            raise ValueError('Native cache/supervision retained-Step mismatch')
        def load(name):
            path=folder/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][name]['SHA256']:
                raise ValueError('Native token cache hash mismatch')
            value=torch.load(path,map_location='cpu',weights_only=True)
            if not torch.isfinite(value['hidden_states']).all():raise ValueError('Nonfinite native token states')
            return value
        raw=load(f'layer_{self.layer}.pt');bags=[]
        for j,span in enumerate(raw['step_token_spans']):
            value=raw
            if j in manifest['causal_override_positions']:
                value=load(f'override_{j}_layer_{self.layer}.pt');span=value['step_token_spans'][0]
            start,end=span;begin=value['token_start']
            if not 0<=start-begin<end-begin<=len(value['hidden_states']):raise ValueError('Invalid token span')
            bags.append(value['hidden_states'][start-begin:end-begin])
        if len(bags)!=len(row['retained_original_step_ids']):raise ValueError('Missing Step token states')
        self.cache[i]=bags
        while len(self.cache)>self.max_cached:self.cache.popitem(last=False)
        return bags

    def bags(self, indices):
        return [self._trajectory(self.positions[int(k)][0])[self.positions[int(k)][1]] for k in indices]
