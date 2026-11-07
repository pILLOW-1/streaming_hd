"""Native model access for offline Answer-method feature construction.

Inputs are frozen token sidecars, never reconstructed from labels or reference
answers. This is a dense attention reference backend: allocation must fit the
complete sequence; no truncation or synthetic attention fallback is allowed.
"""
from __future__ import annotations
import torch
from common.native_attention import attention_blocks


class NativeBackend:
    def __init__(self, path, *, dtype='bfloat16', device='cuda:0'):
        import transformers
        self.device=torch.device(device)
        cfg=transformers.AutoConfig.from_pretrained(path,local_files_only=True)
        self.config=cfg;self.text_config=getattr(cfg,'text_config',cfg)
        self.model=getattr(transformers,cfg.architectures[0]).from_pretrained(path,local_files_only=True,
            dtype=getattr(torch,dtype),attn_implementation='eager',device_map={'':str(device)}).eval()
        self.model.requires_grad_(False)
        modules=[(n,m) for n,m in self.model.named_modules() if n.endswith('layers') and isinstance(m,torch.nn.ModuleList) and len(m)==self.text_config.num_hidden_layers]
        if len(modules)!=1:raise ValueError('Native text decoder cannot be identified uniquely')
        self.norm=self.model.get_submodule(modules[0][0].rsplit('.',1)[0]).norm
        self.head=self.model.get_output_embeddings()
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)

    def logits(self, hidden):
        result=self.head(hidden)
        cap=getattr(self.text_config,'final_logit_softcapping',None)
        if cap is not None:result=torch.tanh(result/cap)*cap
        return result.float()

    @torch.inference_mode()
    def forward(self, ids, *, attention=True):
        tokens=torch.tensor([ids],device=self.device)
        result=self.model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,
                          output_hidden_states=True,output_attentions=attention,logits_to_keep=1)
        states=[h[0] for h in result.hidden_states]
        # Same one-token matrix shape as the native logits_to_keep=1 path.
        if not torch.equal(self.logits(states[-1][-1:]),result.logits[0].float()):
            raise ValueError('Native vocabulary head semantics differ; adapter requires correction')
        weights=attention_blocks(self.config,result.attentions) if attention else {}
        return states,weights

    @torch.inference_mode()
    def step_embeddings(self, states, ids, spans, layer):
        from baselines.rede.kernel import perplexity_weighted_embedding
        values=[]
        for start,end in spans:
            if start<1 or not start<end<=len(ids):raise ValueError('Invalid Step span for next-token probability')
            log_probability=[]
            for begin in range(start,end,128):
                stop=min(end,begin+128)
                logits=self.logits(states[-1][begin-1:stop-1])
                target=torch.tensor(ids[begin:stop],device=self.device)
                log_probability.append(logits.log_softmax(-1).gather(-1,target[:,None]).flatten())
            values.append(perplexity_weighted_embedding(states[layer][start:end],torch.cat(log_probability)).float().cpu())
        return torch.stack(values)

    @torch.inference_mode()
    def filtered_answer(self, native, keep, layer):
        """Remove exact original Step text, retokenize, and re-forward the backbone.

All separators, prompt and final Answer are retained. The sidecar must include
all original Steps in order and the full sequence through the final Answer.
"""
        ids=native['input_ids'];text=self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if self.tokenizer(text,add_special_tokens=False)['input_ids']!=ids:
            raise ValueError('Frozen native full input does not round-trip through tokenizer')
        spans=[];cursor=len(self.tokenizer.decode(ids[:native['reasoning_start_token']],skip_special_tokens=False,clean_up_tokenization_spaces=False))
        for step in native['original_steps']:
            start=text.find(step,cursor)
            if start<0:raise ValueError('Cannot locate original Step in exact native text')
            cursor=start+len(step);spans.append((start,cursor))
        drop=[span for i,span in enumerate(spans) if i not in set(keep)]
        for start,end in reversed(drop):text=text[:start]+text[end:]
        filtered=self.tokenizer(text,add_special_tokens=False)['input_ids']
        if not filtered:raise ValueError('Empty filtered full input')
        states,_=self.forward(filtered,attention=False)
        return states[layer][-1].float().cpu()
