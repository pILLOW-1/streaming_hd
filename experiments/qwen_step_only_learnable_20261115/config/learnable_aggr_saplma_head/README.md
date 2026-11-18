# Learnable aggregation + SAPLMA MLP

Keep the original token scorer architecture and within-Step softmax pooling over frozen Qwen token features. Replace the linear Step head with a 256–128–64 ReLU MLP and a scalar logit. Train the token scorer and MLP together from scratch with the legacy Step-only recipe.

| Training parameter | Value |
| --- | --- |
| Objective | Step BCE with logits |
| Optimization | Adam, learning rate 0.001, weight decay 0.0005, batch size 128 |
| Initialization | Independent per layer, seed 42 |
| Epochs and stopping | At most 20 epochs, legacy patience 5 |

Fit native layers 1–64 independently. Select the layer on VAL by AUROC, then AP, then lower BCE. Freeze the selected checkpoint and VAL F1 threshold before a single TEST prediction pass. Cetus independently recomputes TEST metrics from saved predictions.
