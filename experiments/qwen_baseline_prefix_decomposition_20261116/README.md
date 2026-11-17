# Qwen Prefix decomposition

## Configurations

| Configuration | Design |
| --- | --- |
| [Original direct Streaming](config/direct_streaming/README.md) | Published direct Prefix detector with fixed exponential pooling. |
| [Best published baseline](config/best_published/README.md) | Highest published Prefix AUROC for each domain. |
| [Streaming decomposition, all-position BCE](config/streaming_all_positions/README.md) | Normalized Prefix pooling and BCE on every TRAIN position. |
| [v1](config/v1/README.md) | All-position BCE with legacy Adam and unnormalized Prefix features. |
| [v1 + fixed-aggregation Step](config/v1_fixed_step/README.md) | Use the frozen Step detector trained with fixed aggregation. |
| [v1 + conditional ranking](config/v1_conditional_ranking/README.md) | Add a same-trajectory ranking loss to the conditional logit. |
| [v1 + GRU + memory κ](config/v1_kappa_gru_memory/README.md) | Predict κ with causal GRU memory and compose with frozen Streaming Step scores. |
| [GRU + memory κ with frozen Learnable aggregation](config/gru_memory_kappa_frozen_aggr_repr/README.md) | Feed frozen Learnable aggregation Step representations into GRU memory and compose κ with frozen Step probabilities. |

## Results

Prefix columns report TEST AUROC. Recovery columns report within-trajectory macro AUROC on matched pairs.

| Configuration | Prefix Logic | Prefix Factuality | Recovery VAL Logic | Recovery VAL Factuality | Recovery TEST Logic | Recovery TEST Factuality |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original direct Streaming | 0.8256 | 0.8222 |  |  |  |  |
| Best published baseline | 0.8256 ([Streaming](../streaming_622_p70/result/metrics.json)) | 0.8226 ([Global Mean](../global_mean/622_p70_20261108/result/qwen3_8_27b/Factuality/prefix.json)) |  |  |  |  |
| Streaming decomposition, all-position BCE | 0.8239 | 0.8258 |  |  |  |  |
| v1 | 0.8316 | 0.8181 | 0.563189 | 0.568658 | 0.542960 | 0.552347 |
| v1 + fixed-aggregation Step | 0.8356 | 0.8190 |  |  |  |  |
| v1 + conditional ranking | 0.835883 | 0.816752 | 0.571456 | 0.569815 | 0.555914 | 0.548313 |
| v1 + GRU + memory κ |  |  |  |  |  |  |
| GRU + memory κ with frozen Learnable aggregation | 0.8500 | 0.7890 |  |  | 0.555260 | 0.567283 |
