# Shared Prefix protocol

Compose a frozen Step score with a trainable conditional Prefix head. The table below describes the original SAPLMA and Streaming decomposition recipes. The [v1 recipe](../v1/README.md) specifies its own optimizer and feature normalization.

| Training parameter | Setting |
|---|---|
| Backbone and splits | Frozen Qwen3.8-27B and published 6:2:2 p70 splits |
| Frozen Step probability $`\pi^{\mathrm{S}}_t`$ | Use each baseline's published Step checkpoint. Reuse saved VAL and TEST scores and score TRAIN once. |
| Objective | Use $`y^{\mathrm{P}}_t`$ as the BCE target for $`\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t+(1-\pi^{\mathrm{S}}_t)\,\kappa_t`$ on the configuration's TRAIN positions. Only the conditional Prefix head receives gradients. |
| SAPLMA optimization | Original Glorot initialization and Keras Adam, learning rate 0.001, batch 32, five epochs, no early stopping |
| Streaming optimization | Original LBFGS initialization with L2, then AdamW at 0.001 with weight decay 0.01 for 20 epochs, 64 trajectories per batch and final-position BCE weight 30 |
| Streaming synchronization | The original missed-alarm penalty is zero because the composed probability satisfies $`\pi^{\mathrm{P}}_t \ge \pi^{\mathrm{S}}_t`$. |
| Selection and evaluation | Select from 64 layers using composed Prefix scores on full VAL, and set the threshold on VAL. Evaluate the frozen winner once on TEST. Direct Prefix scores are reused from the published baselines. |
