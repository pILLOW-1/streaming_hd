#!/usr/bin/env python3
"""Pool native decoder hidden states promptly, preserving final norm semantics."""
import gc
import numpy as np
import torch
import transformers


class StopAtLayer(Exception):
    pass


class NativeLayers:
    def __init__(self, model_path, count, width):
        cfg = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.model = getattr(transformers, cfg.architectures[0]).from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
        self.model.requires_grad_(False)
        candidates = [(n, m) for n, m in self.model.named_modules()
                      if n.endswith('layers') and isinstance(m, torch.nn.ModuleList) and len(m) == count]
        assert len(candidates) == 1, [(n, len(m)) for n, m in candidates]
        name, blocks = candidates[0]
        parent = self.model.get_submodule(name.rsplit('.', 1)[0])
        assert hasattr(parent, 'norm'), 'Native final norm must be identified explicitly'
        self.count, self.width = count, width
        self.handles = []
        self.active, self.raw_span, self.pooled, self.raw = set(), None, {}, {}
        self.spans = []
        self.stop, self.stop_layer = False, count
        for layer in range(1, count):
            self.handles.append(blocks[layer-1].register_forward_hook(self._hook(layer)))
        self.handles.append(parent.norm.register_forward_hook(self._hook(count)))

    def _hook(self, layer):
        def hook(module, arguments, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if layer in self.active:
                assert hidden.shape[-1] == self.width
                if self.raw_span is not None:
                    start, end = self.raw_span
                    self.raw[layer] = hidden[0, start:end].detach().cpu()
                if self.spans:
                    values = []
                    for start, end in self.spans:
                        assert 0 <= start < end <= hidden.shape[1]
                        token_values = hidden[0, start:end].float()
                        weights = torch.exp(3 * torch.linspace(0, 1, end-start, device=hidden.device, dtype=torch.float32))
                        weights /= weights.sum()
                        values.append((token_values * weights[:, None]).sum(0).cpu().numpy())
                    self.pooled[layer] = np.stack(values).astype(np.float32)
            if self.stop and layer == self.stop_layer:
                raise StopAtLayer()
        return hook

    @torch.inference_mode()
    def run(self, ids, layers, spans=(), raw_span=None, api=False):
        self.active = set(layers)
        assert self.active and min(self.active) >= 1 and max(self.active) <= self.count
        self.spans, self.raw_span = spans, raw_span
        self.pooled, self.raw = {}, {}
        self.stop, self.stop_layer = not api, max(self.active)
        tensor = torch.tensor([ids], dtype=torch.long, device='cuda')
        try:
            output = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                use_cache=False, output_hidden_states=api, logits_to_keep=1)
            if api:
                for layer in self.active:
                    start, end = raw_span
                    assert torch.equal(output.hidden_states[layer][0, start:end].cpu(), self.raw[layer]), (
                        'native_hidden_states_hook_mismatch', layer)
                del output
        except StopAtLayer:
            assert not api
        assert self.active == set(self.raw if raw_span is not None else self.pooled)
        return self.pooled, self.raw

    def close(self):
        for handle in self.handles:
            handle.remove()
        del self.model
        self.raw, self.pooled = {}, {}
        gc.collect()
        torch.cuda.empty_cache()
