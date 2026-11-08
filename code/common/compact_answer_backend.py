"""Native SDPA hidden states plus the official eager last-query attention.

ReDe consumes only the final Answer token's attention at the last full block.
Keep that actual native query instead of retaining dense attention from every
layer. The unchanged native SDPA function still computes every backbone output.
A short same-Q/K native-eager comparison gates the compact attention path.
"""
import importlib
import torch
from common.native_backend import NativeBackend


class CompactAnswerBackend(NativeBackend):
    def __init__(self,path,*,dtype='bfloat16',device='cuda:0'):
        if dtype!='bfloat16':raise ValueError('This native source requires the frozen BF16 configuration')
        import transformers
        # generation.py is also a script module: resolve this sibling by file,
        # avoiding package shadowing when code/generation is on sys.path.
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('streaming_hd_native_layers',Path(__file__).resolve().parents[1]/'generation/native_layers.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        NativeLayers=native.NativeLayers
        self.config=transformers.AutoConfig.from_pretrained(path,local_files_only=True)
        self.text_config=getattr(self.config,'text_config',self.config)
        self.count=self.text_config.num_hidden_layers
        self.width=self.text_config.hidden_size
        self.device=torch.device(device)
        if str(device) not in ('cuda','cuda:0'):raise ValueError('Use the one PBS-owned visible GPU')
        self.engine=NativeLayers(path,self.count,self.width)
        self.model=self.engine.model;self.head=self.model.get_output_embeddings()
        cls=transformers.PreTrainedTokenizerFast if self.config.model_type=='llama' else transformers.AutoTokenizer
        self.tokenizer=cls.from_pretrained(path,local_files_only=True,use_fast=True)
        types=getattr(self.text_config,'layer_types',None)
        self.last_full=max(i for i in range(self.count) if types is None or types[i] not in ('linear_attention','linear'))
        self.capture=False;self.last_attention=None;self.attention_gate=False;self.hidden_gate=False
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.registry=ALL_ATTENTION_FUNCTIONS
        self.original_sdpa=ALL_ATTENTION_FUNCTIONS['sdpa']
        original=self.original_sdpa
        def compact(module,query,key,value,attention_mask,*args,**kwargs):
            result=original(module,query,key,value,attention_mask,*args,**kwargs)
            if self.capture and getattr(module,'layer_idx',None)==self.last_full:
                eager=getattr(importlib.import_module(type(module).__module__),'eager_attention_forward')
                mask=attention_mask[..., -1:, :] if attention_mask is not None else None
                _,weights=eager(module,query[:,:,-1:,:],key,value,mask,*args,**kwargs)
                if not self.attention_gate:
                    assert query.shape[-2]<=128,'Short native-eager gate must run before a full trajectory'
                    _,reference=eager(module,query,key,value,attention_mask,*args,**kwargs)
                    torch.testing.assert_close(weights,reference[:,:,-1:,:],rtol=1e-3,atol=1e-5)
                    self.attention_gate=True
                self.last_attention=weights[0].detach().float().cpu()
            return result
        ALL_ATTENTION_FUNCTIONS.register('sdpa',compact)

    @torch.inference_mode()
    def forward(self,ids,*,attention=True):
        if not self.hidden_gate:
            probe=ids[:128]
            self.capture=True;self.last_attention=None
            self.engine.run(probe,[self.count],raw_span=(len(probe)-1,len(probe)),api=True)
            assert self.attention_gate and self.last_attention is not None
            self.hidden_gate=True
        self.capture=attention;self.last_attention=None
        _,raw=self.engine.run(ids,[self.count],raw_span=(0,len(ids)))
        states=[None]*self.count+[raw[self.count].to(self.device)]
        weights={}
        if attention:
            assert self.last_attention is not None,'Native full attention module index was not observed'
            assert self.last_attention.shape[-2:]==(1,len(ids))
            weights[self.last_full]=self.last_attention.to(self.device)
        return states,weights

    @torch.inference_mode()
    def step_embeddings(self,states,ids,spans,layer):
        if layer!=self.count:raise ValueError('This compact ReDe configuration declares final-layer Step embeddings')
        return super().step_embeddings(states,ids,spans,layer)

    def close(self):
        self.registry.register('sdpa',self.original_sdpa)
        self.engine.close()

    @torch.inference_mode()
    def filtered_endpoints(self,native,keep,layers):
        from common.native_inputs import exact_decoded_text
        ids=native['input_ids'];text=exact_decoded_text(self.tokenizer,native)
        if self.tokenizer(text,add_special_tokens=False)['input_ids']!=ids:
            raise ValueError('Full canonical input does not round-trip through its pinned tokenizer')
        spans=[];cursor=len(self.tokenizer.decode(ids[:native['reasoning_start_token']],skip_special_tokens=False,clean_up_tokenization_spaces=False))
        for step in native['original_steps']:
            start=text.find(step,cursor)
            if start<0:raise ValueError('Original Step missing from full canonical input')
            cursor=start+len(step);spans.append((start,cursor))
        for start,end in reversed([v for i,v in enumerate(spans) if i not in set(keep)]):
            text=text[:start]+text[end:]
        filtered=self.tokenizer(text,add_special_tokens=False)['input_ids']
        self.capture=False;self.last_attention=None
        _,values=self.engine.run(filtered,layers,raw_span=(len(filtered)-1,len(filtered)))
        return {layer:value[0].float().cpu() for layer,value in values.items()}
