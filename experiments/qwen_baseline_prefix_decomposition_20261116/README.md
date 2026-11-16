# Qwen baseline Prefix decomposition

Test whether composing a frozen Step score with a conditional Prefix head improves Prefix detection over each baseline's published direct Prefix detector on Logic and Factuality. Compare two choices of TRAIN positions for the composed Prefix BCE.

| Configuration | Conditional Prefix readout | TRAIN positions |
|---|---|---|
| SAPLMA decomposition, Step-negative BCE | Endpoint last-token MLP (256–128–64) | $Y_t=0$ |
| SAPLMA decomposition, all-position BCE | Endpoint last-token MLP (256–128–64) | All positions |
| Streaming decomposition, Step-negative BCE | L2-normalized global exponential Prefix pooling and linear probe | $Y_t=0$ |
| Streaming decomposition, all-position BCE | L2-normalized global exponential Prefix pooling and linear probe | All positions |

| Training parameter | Setting |
|---|---|
| Backbone and splits | Frozen Qwen3.8-27B and published 6:2:2 p70 splits |
| Frozen Step score $q_t$ | Use each baseline's published Step checkpoint. Reuse saved VAL and TEST scores and score TRAIN once. |
| Objective | Use $C_t$ as the BCE target for $p_t^C=q_t+(1-q_t)\kappa_t$ on the configuration's TRAIN positions. Only the conditional Prefix head receives gradients. |
| SAPLMA optimization | Original Glorot initialization and Keras Adam, learning rate 0.001, batch 32, five epochs, no early stopping |
| Streaming optimization | Original LBFGS initialization with L2, then AdamW at 0.001 with weight decay 0.01 for 20 epochs, 64 trajectories per batch and final-position BCE weight 30 |
| Streaming synchronization | The original missed-alarm penalty is zero because the composed score satisfies $p_t^C \ge q_t$. |
| Selection and evaluation | Select from 64 layers using composed Prefix scores on full VAL, and set the threshold on VAL. Evaluate the frozen winner once on TEST. Direct Prefix scores are reused from the published baselines. |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| Original direct SAPLMA | 0.8078 | 0.7842 |
| SAPLMA decomposition, Step-negative BCE | 0.7916 | 0.7737 |
| SAPLMA decomposition, all-position BCE | 0.8026 | 0.7846 |
| Original direct Streaming | 0.8256 | 0.8222 |
| Streaming decomposition, Step-negative BCE | 0.8240 | 0.8242 |
| Streaming decomposition, all-position BCE |  | 0.8258 |

### AUROC change from Step-negative training (percentage points)

| Configuration | Logic | Factuality |
|---|---:|---:|
| SAPLMA all-position BCE | +1.10 | +1.09 |
| Streaming all-position BCE |  | +0.15 |

### AUROC change from the original direct Prefix baseline (percentage points)

| Configuration | Logic | Factuality |
|---|---:|---:|
| SAPLMA all-position BCE | -0.52 | +0.03 |
| Streaming all-position BCE |  | +0.36 |

Completed cells are independently verified from saved TEST predictions. Blank cells are pending.
