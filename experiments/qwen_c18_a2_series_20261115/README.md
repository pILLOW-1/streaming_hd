# Qwen C18/A2 readout series

This series examines three related changes to Qwen3.8-27B Step and Prefix detection on Logic and Factuality. The fixed-readout follow-up measures the value of linear history and tests A3 normalization. The learnable A2 runs change the Step readout. The matched stop-gradient runs then isolate the training signal through historical Step representations.

| Stage | Comparison | Results |
|---|---|---|
| [1. Error-memory follow-up](error_memory_followup/README.md) | Fixed A2 versus A1 across VAL layers, plus normalized A3 | Both domains verified |
| [2. Learnable Step readouts](learnable_steps/README.md) | Learn aggregation, representation, or both under A2 | Verified cells below |
| [3. Historical Step stop-gradient](detached_history/README.md) | Same learnable configurations, with gradients also stopped through historical Step representations | Verified cells below |

Stages 2 and 3 use the same trajectory batching, loss, native layers, data split and VAL selection rule. Both detach historical Step probabilities. Stage 3 also detaches the historical vectors in the A2 sum; the current Step readout and history coefficient remain trainable. Fixed-readout references use the earlier training protocol, so cross-stage scores describe the configurations as run.

## Paired VAL check

Fixed A2 beats fixed A1 on Prefix AUROC in 56/64 layers in Logic (median +0.42 pp) and 62/64 layers in Factuality (median +0.54 pp). These are comparisons across candidate layers on the same VAL split.

## Frozen TEST comparison

Each run uses its VAL-selected detector. Recovery AUROC is averaged across eligible trajectories. Empty cells indicate results awaiting independent verification. The [fixed-readout references](../qwen_c18_error_memory_20261114/README.md) come from the preceding A1–A3 experiment.

### Step AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 fixed (C18) | 0.8775 | 0.8820 |
| A2 fixed | 0.8775 | **0.8826** |
| A3 original | **0.8816** | 0.8753 |
| A3 normalized | 0.8775 | 0.8748 |
| A2 learnable · Aggregation |  | 0.8702 |
| A2 learnable · Representation | 0.8681 | 0.8800 |
| A2 learnable · Aggregation + representation | 0.8684 | 0.8766 |
| A2 historical e stop-grad · Aggregation |  | 0.8806 |
| A2 historical e stop-grad · Representation | 0.8662 | 0.8807 |
| A2 historical e stop-grad · Aggregation + representation | 0.8623 | 0.8737 |

### Prefix AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 fixed (C18) | 0.8341 | 0.8106 |
| A2 fixed | 0.8377 | 0.8367 |
| A3 original | 0.8129 | 0.8158 |
| A3 normalized | 0.8320 | 0.8145 |
| A2 learnable · Aggregation |  | 0.8156 |
| A2 learnable · Representation | 0.8335 | **0.8402** |
| A2 learnable · Aggregation + representation | 0.8395 | 0.8252 |
| A2 historical e stop-grad · Aggregation |  | 0.8300 |
| A2 historical e stop-grad · Representation | 0.8292 | 0.8358 |
| A2 historical e stop-grad · Aggregation + representation | **0.8395** | 0.8258 |

### Conditional κ recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 fixed (C18) | 0.5470 | 0.5367 |
| A2 fixed | 0.5583 | 0.5361 |
| A3 original | 0.5363 | **0.5469** |
| A3 normalized | 0.5572 | 0.5202 |
| A2 learnable · Aggregation |  | 0.5333 |
| A2 learnable · Representation | 0.5458 | 0.5418 |
| A2 learnable · Aggregation + representation | 0.5467 | 0.5298 |
| A2 historical e stop-grad · Aggregation |  | 0.5310 |
| A2 historical e stop-grad · Representation | 0.5506 | 0.5321 |
| A2 historical e stop-grad · Aggregation + representation | **0.5585** | 0.5363 |

### Prefix recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 fixed (C18) | 0.5492 | 0.5338 |
| A2 fixed | **0.5636** | 0.5226 |
| A3 original | 0.5360 | 0.5289 |
| A3 normalized | 0.5611 | 0.5266 |
| A2 learnable · Aggregation |  | 0.5263 |
| A2 learnable · Representation | 0.5382 | 0.5288 |
| A2 learnable · Aggregation + representation | 0.5517 | 0.5307 |
| A2 historical e stop-grad · Aggregation |  | 0.5314 |
| A2 historical e stop-grad · Representation | 0.5446 | 0.5334 |
| A2 historical e stop-grad · Aggregation + representation | 0.5613 | **0.5365** |

## Matched change from stopping the historical representation gradient

Values are percentage-point changes from the corresponding learnable A2 configuration to its historical stop-gradient version. Each row uses the same domain and readout.

| Matched comparison | Step | Prefix | κ recovery | Prefix recovery |
|---|---:|---:|---:|---:|
| Logic · Representation | -0.19 | -0.43 | +0.48 | +0.64 |
| Logic · Aggregation + representation | -0.60 | +0.01 | +1.18 | +0.96 |
| Factuality · Aggregation | +1.04 | +1.44 | -0.24 | +0.51 |
| Factuality · Representation | +0.07 | -0.44 | -0.96 | +0.46 |
| Factuality · Aggregation + representation | -0.29 | +0.05 | +0.65 | +0.58 |

## Observations

- Normalizing A3 changes TEST Prefix AUROC by +1.91 pp on Logic and -0.13 pp on Factuality compared with original A3. This change combines history-mean normalization and projection LayerNorm.

- On Factuality, learnable representation has the highest verified overall Prefix AUROC. Stopping the historical representation gradient improves trajectory-averaged Prefix recovery for all three matched readouts, with different directions for their overall Prefix AUROC changes.

- All three Factuality pairs and the Logic representation and joint-readout pairs are verified. The Logic aggregation pair is pending.
