"""Native hidden states and bounded native eager Step-attention means for RHD."""
import importlib,importlib.util
from pathlib import Path
import numpy as np
import torch

class NativeRHD:
    def __init__(self,path,count,width,attention_layers,query_chunk=64):
        spec=importlib.util.spec_from_file_location('rhd_native_layers',Path(__file__).resolve().parents[1]/'generation/native_layers.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.engine=module.NativeLayers(path,count,width);self.model=self.engine.model;self.count=count;self.head=self.model.get_output_embeddings()
        blocks=[(n,m) for n,m in self.model.named_modules() if n.endswith('layers') and isinstance(m,torch.nn.ModuleList) and len(m)==count];assert len(blocks)==1
        self.norm=self.model.get_submodule(blocks[0][0].rsplit('.',1)[0]).norm;self.config=getattr(self.model.config,'text_config',self.model.config)
        self.attention_layers=set(attention_layers);self.chunk=query_chunk;self.spans=[];self.matrices={};self.gated=set()
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.registry=ALL_ATTENTION_FUNCTIONS;self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];original=self.original
        def wrapped(module,query,key,value,attention_mask,*args,**kwargs):
            output=original(module,query,key,value,attention_mask,*args,**kwargs);layer=getattr(module,'layer_idx',None)
            if layer not in self.attention_layers or not self.spans:return output
            assert query.shape[0]==1 and query.shape[-2]==key.shape[-2]
            size=query.shape[-2];spans=self.spans;n=len(spans);start_index=int(np.floor(n*.8));matrix=np.zeros((n,n),dtype=np.float64)
            eager=getattr(importlib.import_module(type(module).__module__),'eager_attention_forward')
            def mask_for(a,b):
                if attention_mask is not None:return attention_mask[...,a:b,:]
                q=torch.arange(a,b,device=query.device)[:,None];k=torch.arange(size,device=query.device)[None,:];allowed=k<=q;window=kwargs.get('sliding_window',getattr(module,'sliding_window',None))
                if window is not None and window>0:allowed=allowed & (k>q-window)
                return torch.zeros((1,1,b-a,size),dtype=query.dtype,device=query.device).masked_fill(~allowed[None,None],torch.finfo(query.dtype).min)
            for i in range(start_index,n):
                left,right=spans[i];assert 0<=left<right<=size
                for a in range(left,right,self.chunk):
                    b=min(right,a+self.chunk);_,weights=eager(module,query[:,:,a:b],key,value,mask_for(a,b),*args,**kwargs)
                    for j,(lo,hi) in enumerate(spans[:start_index]):
                        piece=weights[0,:, :,lo:hi].float();assert torch.isfinite(piece).all()
                        matrix[i,j]+=float(piece.double().sum().cpu())/(query.shape[1]*(right-left)*(hi-lo))
                    del weights
            if layer not in self.gated:
                assert size<=128,'Short native eager reference must run first'
                _,reference=eager(module,query,key,value,mask_for(0,size),*args,**kwargs)
                for i in range(start_index,n):
                    a,b=spans[i]
                    for j,(lo,hi) in enumerate(spans[:start_index]):np.testing.assert_allclose(matrix[i,j],float(reference[0,:,a:b,lo:hi].float().mean()),rtol=1e-5,atol=1e-7)
                self.gated.add(layer)
            self.matrices[layer]=matrix;return output
        self.registry.register('sdpa',wrapped)
    def logits(self,hidden,layer):
        # The backbone's extraction hooks must not observe a separate vocabulary
        # projection: its native final norm is reused here on a two-dimensional
        # token chunk after the full causal forward has already completed.
        previous=(self.engine.active,self.engine.spans,self.engine.raw_span,self.engine.stop)
        self.engine.active,self.engine.spans,self.engine.raw_span,self.engine.stop=set(),[],None,False
        try:
            value=hidden.to('cuda:0')
            if layer!=self.count:value=self.norm(value)
            logits=self.head(value);cap=getattr(self.config,'final_logit_softcapping',None)
            if cap is not None:logits=torch.tanh(logits/cap)*cap
            return logits.float()
        finally:
            self.engine.active,self.engine.spans,self.engine.raw_span,self.engine.stop=previous
    def forward(self,ids,layers,spans,api=False):
        self.spans=spans;self.matrices={};_,raw=self.engine.run(ids,layers,raw_span=(0,len(ids)),api=api)
        return raw,dict(self.matrices)
    def close(self):self.registry.register('sdpa',self.original);self.engine.close()
