# A2 readout exploration

Change the Step representation and memory content while retaining the original A2 Prefix readout, objective and joint VAL selection. Accepted results are maintained across five domains; unfinished cells remain empty. Original A2 also includes the verified Science and Multi-hop extensions.

| Configuration | Design |
|---|---|
| [R0: original A2](../README.md#results) | Step exponential pool for Step prediction and memory; reuse original results. |
| [R1: last token](config/r1_r2/README.md) | Last Step token for Step prediction and memory. |
| [R2: exponential pool + last token](config/r1_r2/README.md) | Concatenate the two Step readouts for prediction and memory. |
| [H1: HaMI Step + complete memory](config/hami_topk_memory/README.md) | Top-k mean token error probability; original Step exponential pool as memory. |
| [H2: HaMI Step + Top-k memory](config/hami_topk_memory/README.md) | Same scoring architecture; mean raw Top-k token features as memory. |

## Step TEST AUROC

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
|---|---:|---:|---:|---:|---:|
| Streaming | 0.8716 | 0.8720 | 0.8145 | 0.7852 | 0.8953 |
| Best published baseline | 0.8875 (HaMI) | 0.8735 (HaMI) | **0.8255** (SAPLMA) | **0.8046** (SAPLMA) | 0.9080 (HaMI) |
| R0: original A2 | 0.8775 | **0.8826** | 0.8218 |  | **0.9128** |
| R1: last token | 0.8822 | 0.8689 |  |  |  |
| R2: exponential pool + last token | 0.8766 | 0.8779 |  |  |  |
| H1: HaMI Step + complete memory | 0.8838 | 0.8723 |  |  |  |
| H2: HaMI Step + Top-k memory | **0.8897** | 0.8723 |  |  |  |
## Prefix TEST AUROC

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
|---|---:|---:|---:|---:|---:|
| Streaming | 0.8256 | 0.8222 | 0.7616 | 0.7357 | 0.8344 |
| Best published baseline | 0.8256 (Streaming) | 0.8226 (Global Mean) | 0.7644 (Global Mean) | **0.7398** (SAPLMA) | 0.8381 (Global Mean) |
| R0: original A2 | 0.8377 | 0.8367 | **0.7650** |  | **0.8494** |
| R1: last token | 0.8439 | **0.8376** |  |  |  |
| R2: exponential pool + last token | **0.8445** | **0.8376** |  |  |  |
| H1: HaMI Step + complete memory | 0.8356 | 0.8309 |  |  |  |
| H2: HaMI Step + Top-k memory | 0.8376 | 0.8295 |  |  |  |
## Inherited vs. recovered TEST AUROC

Pool all inherited TEST positives across trajectories: Step clean, Prefix erroneous. Negatives are recovered positions: Step and Prefix clean, with an earlier retained erroneous Prefix. Rank by the final Prefix score. This is not a within-trajectory average.

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
|---|---:|---:|---:|---:|---:|
| Streaming | 0.7020 | 0.7205 | 0.7333 | **0.7334** | 0.7564 |
| Best published baseline | 0.7020 (Streaming) | 0.7233 (Global Mean) | **0.7381** (Global Mean) | 0.7258 (SAPLMA) | 0.7639 (Global Mean) |
| R0: original A2 | 0.7204 | 0.7476 | 0.7363 |  | **0.7660** |
| R1: last token | 0.7286 | 0.7525 |  |  |  |
| R2: exponential pool + last token | **0.7344** | **0.7545** |  |  |  |
| H1: HaMI Step + complete memory | 0.7116 | 0.7332 |  |  |  |
| H2: HaMI Step + Top-k memory | 0.7143 | 0.7308 |  |  |  |

Bold marks the best displayed four-decimal value per column. [Baseline references](result/baseline_references.json) reuse the published main comparison; the best baseline is selected separately per domain and detection level, with method names shown. [Readout results](result/index.json) retain exact metrics and provenance. The [original Multi-hop A2 result](../k_design/result/Multi-hop/A2.json) uses joint VAL selection (L49, epoch 3), one frozen TEST and independent saved-prediction CPU acceptance. The [Science original A2 result](result/Science/R0.json) selects L46, epoch 1; its [configuration](config/original_a2_science_20261119.json) and independent CPU acceptance cover 46,175 TEST positions. No model inference was repeated.

Streaming and best-baseline recovery scores reuse the [verified five-domain diagnostic](../../prefix_recovery_diagnostic_20261112/README.md). Best baseline is defined by overall Prefix TEST AUROC for that domain; the method is named in each cell. See [exact recovery references and prediction hashes](result/recovery_baseline_references.json). No inference or new model selection was performed.
H1/H2 use the same HaMI token-scoring architecture and C18 joint objective, with independently trained parameters. Factuality selects L50/epoch1 for both; Logic selects L43/epoch3 (H1) and L43/epoch4 (H2). Both domains passed independent saved-prediction CPU verification. [Design, training settings and source](config/hami_topk_memory/README.md).
