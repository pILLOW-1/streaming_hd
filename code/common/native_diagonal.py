"""Memory-bounded native eager causal diagonals alongside unchanged native SDPA."""
import importlib
import torch


class NativeDiagonal:
    def __init__(self,eligible,query_chunk=128):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.registry=ALL_ATTENTION_FUNCTIONS;self.original=ALL_ATTENTION_FUNCTIONS['sdpa']
        self.eligible=set(eligible);self.chunk=query_chunk;self.values={};self.gated=set()
        original=self.original
        def wrapped(module,query,key,value,attention_mask,*args,**kwargs):
            output=original(module,query,key,value,attention_mask,*args,**kwargs)
            layer=getattr(module,'layer_idx',None)
            if layer not in self.eligible:return output
            assert query.shape[0]==1 and query.shape[-2]==key.shape[-2], 'Fresh full causal sequence required'
            if attention_mask is not None and not isinstance(attention_mask,torch.Tensor):raise ValueError('Unsupported native attention mask type')
            eager=getattr(importlib.import_module(type(module).__module__),'eager_attention_forward')
            size=query.shape[-2];parts=[]
            def mask_for(start,end):
                if attention_mask is not None:return attention_mask[...,start:end,:]
                q=torch.arange(start,end,device=query.device)[:,None];k=torch.arange(size,device=query.device)[None,:]
                allowed=k<=q
                window=kwargs.get('sliding_window',getattr(module,'sliding_window',None))
                if window is not None:allowed=allowed & (k>q-window)
                mask=torch.zeros((1,1,end-start,size),dtype=query.dtype,device=query.device)
                return mask.masked_fill(~allowed[None,None],torch.finfo(query.dtype).min)
            for start in range(0,size,self.chunk):
                end=min(size,start+self.chunk);mask=mask_for(start,end)
                _,weights=eager(module,query[:,:,start:end],key,value,mask,*args,**kwargs)
                rows=torch.arange(end-start,device=query.device);columns=torch.arange(start,end,device=query.device)
                diagonal=weights[0,:,rows,columns]
                assert torch.isfinite(diagonal).all() and (diagonal>0).all(),'LLM-Check native diagonal is undefined; do not clamp or drop'
                parts.append(diagonal.detach().cpu())
                del weights
            diagonal=torch.cat(parts,dim=-1)
            if layer not in self.gated:
                assert size<=128,'Short native-eager comparison required before production'
                _,reference=eager(module,query,key,value,mask_for(0,size),*args,**kwargs)
                torch.testing.assert_close(diagonal.to(reference.device),reference[0].diagonal(dim1=-2,dim2=-1),rtol=1e-3,atol=1e-5)
                self.gated.add(layer)
            self.values[layer]=diagonal
            return output
        ALL_ATTENTION_FUNCTIONS.register('sdpa',wrapped)
    def clear(self):self.values={}
    def close(self):self.registry.register('sdpa',self.original)
