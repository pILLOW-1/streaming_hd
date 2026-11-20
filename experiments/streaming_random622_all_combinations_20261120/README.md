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
| Original Streaming — AUROC (%) | **78.20** | **77.14** | | | |
| Original Streaming — AP (%) | **60.82** | **73.76** | | | |

Published completed results are Qwen Science and Math, each at Step and Prefix levels. Previously published Logic, Fact and Multi-hop comparisons remain in the [random-split comparison](../../reports/main_result/comparison_random_split_candidate.md). Full-precision metrics, selected layers/updates, frozen VAL thresholds and verification hashes are in [the result index](result/index.json). No samples, hidden states, trained weights or saved predictions are included.

The Math Step configuration is recorded separately in [config_math_step.json](config_math_step.json); its detector uses the same original model-specific recipe.

The Math Prefix configuration is recorded in [config_math_prefix.json](config_math_prefix.json). Its Step teacher is the independently selected Math Step checkpoint; no Step fitting or TEST was repeated for this continuation.

## Additional accepted results

[All accepted cells](result/accepted_cells.json) includes Gemma Logic, Factuality, Science and Multi-hop Step/Prefix results, the accepted Qwen Step/Prefix results, and nine Streaming Answer results. The candidate layers are 32 for DeepSeek, 60 for Gemma and 64 for Qwen; the training table above describes the Qwen recipe. See [the model-specific configuration](config_all_models.json). No unfinished DeepSeek or Gemma Math result is included.

Answer uses the last retained Prefix score from the frozen Prefix detector; it does not select an Answer layer or checkpoint. Seven Answer cells have their own frozen VAL F1max thresholds. Qwen Logic and Factuality have independently verified AUROC, AP and TPR@1%FPR only: complete saved selected Prefix VAL predictions are unavailable locally, so ACC/F1 and thresholds remain unavailable. No detector inference was repeated.

Gemma Science Step had two same-layer update artifacts with identical parameter bytes and identical VAL metrics. They represent the same detector; artifact identity was resolved before TEST without preferring an earlier update or changing model selection. The equivalence-amendment hash is retained with the result.
