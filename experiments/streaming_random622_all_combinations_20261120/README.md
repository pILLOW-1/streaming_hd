# Original Streaming on the random 6:2:2 split

Assess the original Streaming detector on an unconstrained question-family random split. This is an exploratory re-split of historically inspected trajectories; the original p70 split remains a separate experiment.

| Configuration | Step | Prefix |
| --- | --- | --- |
| Original Streaming | Fixed exponential current-Step pool, L2 normalization, linear detector | Fixed exponential visible-reasoning pool, L2 normalization, linear detector with the newly trained Step teacher |

| Training parameter | Step | Prefix |
| --- | --- | --- |
| Seed / candidate layers | 42 / all 64 native Qwen layers | 42 / all 64 native Qwen layers |
| Optimizer | Original L-BFGS, lr 1, max 100, strong Wolfe | Original L-BFGS initialization; AdamW lr 0.001, weight decay 0.01 |
| Training budget | Original L-BFGS call | 20 epochs, 64 trajectories per batch |
| Objective | Regularized BCE, coefficient 0.0001 | Original anchor/sync objective, coefficients 30/120 |
| Selection | Complete own VAL AUROC, then AP | Complete own VAL AUROC, then AP |
| Decision threshold | Complete VAL F1 maximum | Complete VAL F1 maximum |
| TEST | Selected checkpoint once, independent CPU verification of saved predictions | Same |

## Step TEST results

| Configuration | Math | Science | Logic | Multi-hop | Fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Streaming — AUROC (%) | **78.93** | **82.09** | | | |
| Original Streaming — AP (%) | **21.79** | **24.91** | | | |

## Prefix TEST results

| Configuration | Math | Science | Logic | Multi-hop | Fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Streaming — AUROC (%) | | **77.14** | | | |
| Original Streaming — AP (%) | | **73.76** | | | |

Published completed results are Qwen Science Step/Prefix and Qwen Math Step. Math Prefix has no complete TEST and remains blank. Previously published Logic, Fact and Multi-hop comparisons remain in the [random-split comparison](../../reports/main_result/comparison_random_split_candidate.md). Full-precision metrics, selected layers/updates, frozen VAL thresholds and verification hashes are in [the result index](result/index.json). No samples, hidden states, trained weights or saved predictions are included.

The Math Step configuration is recorded separately in [config_math_step.json](config_math_step.json); its detector uses the same original model-specific recipe.
