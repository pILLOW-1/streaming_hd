# Standalone Qwen Step detectors

## Configurations

| Configuration | Design |
| --- | --- |
| [Original Streaming](config/original_streaming/README.md) | Published direct Step detector using fixed positional weights. |
| [Best published baseline](config/best_published/README.md) | Highest published Step AUROC per domain. |
| [Streaming fixed aggregation](config/streaming_fixed/README.md) | Pool frozen token features with fixed exponential weights. |
| [Learnable aggregation](config/learnable_aggregation/README.md) | Learn weights from token content and pool frozen token features. |
| [Learnable aggregation + SAPLMA MLP](config/learnable_aggr_saplma_head/README.md) | Train the token scorer jointly with a SAPLMA-style Step head. |
| [Learnable aggregation, hard Top-10%](config/hard_top10/README.md) | Mask the remaining logits to −∞ before one within-Step softmax. |
| [Four-head attention + gated residual](config/fourhead_gated/README.md) | Pool each Step using four token-attention heads and a gated mean residual. |
| [Top-3 equal-weight logit fusion](config/frozen_top3_equal/README.md) | Average logits from three frozen Learnable aggregation detectors selected by Step VAL AUROC. |

### Legacy recipe Step TEST AUROC

Rows marked “trajectory recipe” use trajectory batching. Empty cells have no verified result.

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
| --- | --- | ---: | ---: | ---: | ---: |
| [Original Streaming](../streaming_622_p70/result/metrics.json) | 0.8716 | 0.8720 | 0.8145 | 0.7852 | 0.8953 |
| Best published baseline | 0.8875 ([HaMI](../hami/622_p70_20261108/result/qwen3_8_27b/Logic/step.json)) | 0.8735 ([HaMI](../hami/622_p70_20261108/result/qwen3_8_27b/Factuality/step.json)) | 0.8255 ([SAPLMA](../saplma/622_p70_20261108/result/qwen3_8_27b/Science/step.json)) | 0.8046 ([SAPLMA](../saplma/622_p70_20261108/result/qwen3_8_27b/Math/step.json)) | 0.9080 ([HaMI](../hami/622_p70_20261108/result/qwen3_8_27b/Multi-hop/step.json)) |
| Streaming fixed aggregation (legacy recipe) | 0.8736 | 0.8808 | 0.8203 |  | 0.9087 |
| Learnable aggregation | 0.8871 | 0.8816 | 0.8219 | 0.7932 | 0.9177 |
| Learnable aggregation + SAPLMA MLP | [0.8919](result/legacy_recipe/Logic/learnable_aggr_saplma_head.json) | [0.8803](result/legacy_recipe/Factuality/learnable_aggr_saplma_head.json) |  |  |  |
| Learnable aggregation, hard Top-10% | 0.856127 | 0.859060 |  |  |  |
| Four-head attention + gated residual | 0.877537 | 0.876701 |  |  |  |
| Top-3 equal-weight logit fusion | 0.8941 | 0.8829 | 0.8309 |  | 0.9230 |
