# Qwen baseline Prefix decomposition

Test whether composing a frozen Step score with a conditional Prefix head improves Prefix detection over each baseline's published direct Prefix detector on Logic and Factuality. Compare two choices of TRAIN positions for the composed Prefix BCE.

| Configuration | Conditional Prefix readout | TRAIN positions |
|---|---|---|
| SAPLMA decomposition, Step-negative BCE | Endpoint last-token MLP (256–128–64) | $`y^{\mathrm{S}}_t=0`$ |
| SAPLMA decomposition, all-position BCE | Endpoint last-token MLP (256–128–64) | All positions |
| Streaming decomposition, Step-negative BCE | L2-normalized global exponential Prefix pooling and linear probe | $`y^{\mathrm{S}}_t=0`$ |
| Streaming decomposition, all-position BCE | L2-normalized global exponential Prefix pooling and linear probe | All positions |

| Training parameter | Setting |
|---|---|
| Backbone and splits | Frozen Qwen3.8-27B and published 6:2:2 p70 splits |
| Frozen Step probability $`\pi^{\mathrm{S}}_t`$ | Use each baseline's published Step checkpoint. Reuse saved VAL and TEST scores and score TRAIN once. |
| Objective | Use $`y^{\mathrm{P}}_t`$ as the BCE target for $`\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t+(1-\pi^{\mathrm{S}}_t)\,\kappa_t`$ on the configuration's TRAIN positions. Only the conditional Prefix head receives gradients. |
| SAPLMA optimization | Original Glorot initialization and Keras Adam, learning rate 0.001, batch 32, five epochs, no early stopping |
| Streaming optimization | Original LBFGS initialization with L2, then AdamW at 0.001 with weight decay 0.01 for 20 epochs, 64 trajectories per batch and final-position BCE weight 30 |
| Streaming synchronization | The original missed-alarm penalty is zero because the composed probability satisfies $`\pi^{\mathrm{P}}_t \ge \pi^{\mathrm{S}}_t`$. |
| Selection and evaluation | Select from 64 layers using composed Prefix scores on full VAL, and set the threshold on VAL. Evaluate the frozen winner once on TEST. Direct Prefix scores are reused from the published baselines. |

**Legacy Streaming variant.** Keep the published Streaming Step probabilities frozen and use unnormalized exponential Prefix features. Train the linear conditional head with mean composed Prefix BCE on all TRAIN positions, using seed 42, random batches of 128 positions, Adam at 0.001 with weight decay 0.0005, up to 20 epochs, and patience 5. Use the legacy plateau schedule. Rank candidates by VAL Prefix AUROC, AP, then lower BCE. Evaluate the frozen winner once on TEST.

**Fixed-aggregation Step variant.** Keep the legacy Prefix recipe and replace the frozen Step probabilities with the independently VAL-selected legacy fixed-aggregation Step detector for each domain. Its fixed weights are proportional to exp(3u) within each Step. Reuse its saved VAL and TEST predictions and score TRAIN once from the frozen checkpoint.

## Prefix TEST AUROC

Future Prefix experiments are reported in this table.

| Configuration | Logic | Factuality |
|---|---:|---:|
| Original direct Streaming | 0.8256 | 0.8222 |
| Best published baseline | 0.8256 ([Streaming](../streaming_622_p70/result/metrics.json)) | 0.8226 ([Global Mean](../global_mean/622_p70_20261108/result/qwen3_8_27b/Factuality/prefix.json)) |
| Streaming decomposition, all-position BCE | 0.8239 | 0.8258 |
| Streaming decomposition, all-position BCE (legacy optimizer, unnormalized features) | 0.8316 | 0.8181 |
| Streaming decomposition, all-position BCE (legacy optimizer, fixed-aggregation Step) | 0.8356 | 0.8190 |

### Other Prefix comparisons

| Configuration | Logic | Factuality |
|---|---:|---:|
| Original direct SAPLMA | 0.8078 | 0.7842 |
| SAPLMA decomposition, Step-negative BCE | 0.7916 | 0.7737 |
| SAPLMA decomposition, all-position BCE | 0.8026 | 0.7846 |
| Streaming decomposition, Step-negative BCE | 0.8240 | 0.8242 |
