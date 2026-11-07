"""Production Streaming local_exp/global_exp pooling, without scheduler dependencies."""
import numpy as np
import torch
from native_layers import NativeLayers, StopAtLayer

class BothPools(NativeLayers):
    def __init__(self,*a,**kw):
        self.global_start=0;self.local={}
        super().__init__(*a,**kw)

    def _hook(self,layer):
        def hook(module,arguments,output):
            h=output[0] if isinstance(output,(tuple,list)) else output
            if layer in self.active:
                if self.raw_span is not None:
                    a,b=self.raw_span;self.raw[layer]=h[0,a:b].detach().cpu()
                if self.spans:
                    if layer in getattr(self,'global_layers',self.active):
                        start=self.global_start;last=max(e for _,e in self.spans);assert 0<=start<last<=h.shape[1] and .003*(last-start)<600
                        values=h[0,start:last].float().cpu().numpy();weight=np.exp(.003*np.arange(1,last-start+1,dtype=np.float64));num=np.zeros(self.width,dtype=np.float64);den=np.float64(0);previous=0;global_values=[]
                        for a,b in self.spans:
                            end=b-start;assert end>=previous
                            # Exact official per-Step FP64 segment accumulation.
                            if end>previous:
                                num+=(values[previous:end]*weight[previous:end,None]).sum(axis=0);den+=weight[previous:end].sum()
                            global_values.append((num/den).astype(np.float32));previous=end
                        self.pooled[layer]=np.stack(global_values)
                    else:self.pooled[layer]=None  # Active local-only layer, no repeated global pooling.
                    if layer in getattr(self,'local_layers',self.active):
                        local=[]
                        for a,b in self.spans:
                            v=h[0,a:b].float();w=torch.exp(3*torch.linspace(0,1,b-a,device=h.device,dtype=torch.float32));w/=w.sum();local.append((v*w[:,None]).sum(0).cpu().numpy())
                        self.local[layer]=np.stack(local).astype(np.float32)
            if self.stop and layer==self.stop_layer:raise StopAtLayer()
        return hook

    def both(self,row,layers,start):
        self.global_start=start;self.local={};pools,_=self.run(row['input_ids'],layers,spans=row['step_token_spans']);return {layer:value for layer,value in pools.items() if value is not None},self.local
