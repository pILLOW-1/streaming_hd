import numpy as np
import torch

def pools(visible, spans, reasoning_start, methods):
    """Raw FP32 pools, same local-exp/global-exp equations as Streaming."""
    result = {m: [] for m in methods}
    acc = np.zeros(visible.shape[1], dtype=np.float64)
    weighted = acc.copy(); denominator = 0.0; previous = 0
    for a, b in spans:
        left, end = a-reasoning_start, b-reasoning_start
        assert 0 <= left < end <= len(visible)
        if 'last_token' in result: result['last_token'].append(visible[end-1])
        if 'step_exp' in result:
            # Match torch.float32 Streaming local weighting exactly.
            bag = torch.from_numpy(visible[left:end])
            weight = torch.exp(3*torch.linspace(0, 1, len(bag), dtype=torch.float32))
            weight /= weight.sum()
            result['step_exp'].append((bag*weight[:, None]).sum(0).numpy())
        if 'global_mean' in result:
            acc += visible[previous:end].sum(0, dtype=np.float64)
            result['global_mean'].append((acc/end).astype(np.float32))
        if 'prefix_exp' in result:
            # Rescale by the current endpoint for stable exp at arbitrary native lengths.
            # This is algebraically the same normalized exp(.003*j), not truncation.
            scale = np.exp(-0.003*(end-previous))
            weighted *= scale; denominator *= scale
            for lo in range(previous, end, 1024):
                hi = min(end, lo+1024)
                weight = np.exp(0.003*(np.arange(lo, hi, dtype=np.float64)+1-end))
                weighted += np.einsum('ij,i->j', visible[lo:hi], weight, dtype=np.float64)
                denominator += weight.sum()
            result['prefix_exp'].append((weighted/denominator).astype(np.float32))
        previous = end
    return {k: np.stack(v).astype(np.float32) for k, v in result.items()}
