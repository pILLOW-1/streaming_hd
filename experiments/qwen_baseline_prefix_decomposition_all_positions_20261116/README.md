# Qwen Prefix decomposition with all-position BCE

Compare all-position supervision with the [Step-negative-only decomposition](../qwen_baseline_prefix_decomposition_20261116/README.md) on Logic and Factuality.

| Configuration | Frozen Step scorer | Conditional Prefix scorer |
|---|---|---|
| SAPLMA all-position BCE | Published SAPLMA | Native endpoint MLP |
| Streaming all-position BCE | Published Streaming | Global exponential pooling and linear probe |

| Training parameter | Setting |
|---|---|
| Data and predictions | Original Qwen3.8-27B 6:2:2 p70 splits and frozen Step predictions for TRAIN, VAL and TEST |
| Objective | Composed Prefix BCE on every TRAIN position with target $C_t$ and score $p_t^C=q_t+(1-q_t)\kappa_t$. Gradients update only the conditional head. |
| Baseline recipe | Same network, features, optimizer, batch construction, epoch limits and weights as the Step-negative-only experiment |
| Selection and evaluation | Select from all 64 layers using composed Prefix VAL AUROC, then AP. Evaluate the frozen winner once on TEST. |

## Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| SAPLMA all-position BCE | 0.8822 | 0.8607 |
| Streaming all-position BCE | 0.8716 | 0.8720 |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| SAPLMA all-position BCE | | |
| Streaming all-position BCE | | |

The original Step scores are reused. Prefix cells are filled after independent verification from saved TEST predictions.
