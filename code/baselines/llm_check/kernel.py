"""Official attention-score statistic; causal diagonal only, no fake attention.

Source common_utils.get_attn_eig_prod at 2f3bf9e: SUM over heads of the mean
log diagonal. Prompt inclusion is the existing project's full-prefix adaptation.
Score orientation is bound separately on TRAIN; it is never chosen on TEST.
"""
import torch


def attention_score(diagonal):
    if diagonal.ndim != 2 or not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise ValueError("Strictly positive native [heads, visible_tokens] diagonal required")
    return diagonal.double().log().mean(dim=-1).sum()
