# Qwen baseline Prefix decomposition

Test whether a frozen Step detector and a conditional Prefix head improve Prefix detection over each baseline's published direct Prefix detector on the same Logic and Factuality splits.

| Configuration | Frozen Step score | Conditional head |
|---|---|---|
| SAPLMA decomposition | Published SAPLMA Step checkpoint | Last contextual token at the current endpoint, MLP 256–128–64 |
| Streaming decomposition | Published Streaming Step checkpoint | L2-normalized global exponential pooling over the visible generated Prefix, linear probe |

| Training parameter | Setting |
|---|---|
| Backbone and splits | Frozen Qwen3.8-27B and the published 6:2:2 p70 split for each domain |
| Step score inputs | Reuse saved VAL and TEST scores, and score TRAIN once with the same frozen checkpoint |
| Objective | On TRAIN positions with Step label $Y_t=0$, apply the Prefix BCE to the composed probability $p_t^C=q_t+(1-q_t)\kappa_t$ with target $C_t$. Only the conditional head receives gradients. |
| SAPLMA optimization | Original Glorot initialization and Keras Adam, learning rate 0.001, batch 32, five epochs, no early stopping |
| Streaming optimization | Original LBFGS initialization with L2, then AdamW at 0.001 with weight decay 0.01 for 20 epochs, 64 trajectories per batch and final-position BCE weight 30 |
| Streaming synchronization | Its original missed-alarm penalty is zero for the composed score, which satisfies $p_t^C \ge q_t$ by construction. |
| Selection and evaluation | Use composed Prefix scores on the baseline's full VAL cohort for layer selection and thresholding. Evaluate the frozen winner once on TEST. Reuse published direct Prefix results. |

## Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| SAPLMA decomposition | 0.8822 | 0.8607 |
| Streaming decomposition | 0.8716 | 0.8720 |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Original direct SAPLMA | 0.8078 | 0.7842 |
| SAPLMA decomposition | | |
| Original direct Streaming | 0.8256 | 0.8222 |
| Streaming decomposition | | |

**Status.** Locating and validating baseline artifacts for RunPod. Results will be entered after independent verification from saved predictions.
