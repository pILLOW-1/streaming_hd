# A2 readout exploration

Change the Step representation and memory content while retaining the original A2 Prefix readout, objective and joint VAL selection. Factuality results are independently verified; unfinished Logic cells remain empty.

| Configuration | Design |
|---|---|
| [R0: original A2](../README.md#results) | Step exponential pool for Step prediction and memory; reuse original results. |
| [R1: last token](config/r1_r2/README.md) | Last Step token for Step prediction and memory. |
| [R2: exponential pool + last token](config/r1_r2/README.md) | Concatenate the two Step readouts for prediction and memory. |

## Step TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| R0: original A2 | 0.8775 | **0.8826** |
| R1: last token | | 0.8689 |
| R2: exponential pool + last token | | 0.8779 |

## Prefix TEST AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| R0: original A2 | 0.8377 | 0.8367 |
| R1: last token | | **0.8376** |
| R2: exponential pool + last token | | **0.8376** |

## Inherited vs. recovered TEST AUROC

Pool all inherited TEST positives across trajectories: Step clean, Prefix erroneous. Negatives are recovered positions: Step and Prefix clean, with an earlier retained erroneous Prefix. Rank by the final Prefix score. This is not a within-trajectory average.

| Configuration | Logic | Factuality |
|---|---:|---:|
| R0: original A2 | 0.7204 | 0.7476 |
| R1: last token | | 0.7525 |
| R2: exponential pool + last token | | **0.7545** |

Bold marks the best displayed four-decimal value in each column. Exact values, selected layer/epoch, source bindings and prediction hashes are in the [result index](result/index.json). Original A2 references come from the parent report and its pooled recovery analysis. No original A2 fit or TEST inference was repeated.
