# Prefix detectors track whether an error occurred, not whether it is still active

## Motivation

The [notation document](../../theory/STEP_PREFIX_NOTATION.md) writes Prefix error probability as

```math
\pi^C_t=\pi^Y_t+(1-\pi^Y_t)\kappa_t,\qquad \kappa_t=P(C_t=1\mid Y_t=0, X_{\le t}).
```

The [composition analysis](../step_prefix_composition_20261111/README.md) shows that about 85% of Prefix positives are inherited (Y=0, C=1). Prefix detection is therefore mostly a matter of κ: when the current Step adds no new error, is an earlier error still active?

A detector can approximate this with a weaker cue: whether any error has occurred earlier in the trajectory. Both cues agree before the first error. They disagree only after an error has been corrected. We use those positions to test which cue a detector actually uses.

## Analysis settings

On VAL positions with Y=0, the positive class is **inherited** (C=1). Negatives are split by history:

| Negative | Definition | What separates it from inherited |
|---|---|---|
| fresh clean | C=0, no earlier C=1 in the trajectory | "an error occurred" suffices |
| recovered clean | C=0, some earlier C=1 in the trajectory | requires "the error is still active" |

- **AUROC_fresh:** inherited vs fresh clean.
- **AUROC_rec:** inherited vs recovered clean.
- **gap:** AUROC_fresh − AUROC_rec.

A detector that tracks κ itself has a small gap. A detector that tracks only "an error occurred" has a high AUROC_fresh and a low AUROC_rec.

| Setting | Value |
|---|---|
| Dataset | Retained Streaming HD, 6:2:2; **VAL only** |
| Scope | Three generators, five domains |
| Scores | Each detector's frozen, selected-layer VAL Prefix score saved by its original run; no retraining or inference |
| Detectors | Streaming, SAPLMA, global mean, LLM-Check, ICR Probe |
| Integrity checks | Saved position ids and labels equal the dataset loader's in order. Recomputed VAL Prefix AUROC equals the value recorded at selection. Published Streaming cells are matched to their source runs by freeze SHA256. |
| Uncertainty | 95% percentile intervals from 1,000 trajectory-bootstrap replicates |

TEST is excluded because these statistics inform method design.

ICR Probe covers 7 of 15 combinations; its run has not finished in the other 8. Those cells are listed as pending in `result/recovery.json`.

## Findings

### 1. Every detector separates inherited errors from fresh clean positions far better than from recovered ones

| Detector | Combinations | AUROC_rec | AUROC_fresh | gap (range) | gap CI above 0 |
|---|---:|---:|---:|---:|---:|
| Streaming | 15 | 0.764 | 0.976 | 0.212 (0.026–0.365) | 15 |
| SAPLMA | 15 | 0.749 | 0.957 | 0.208 (0.020–0.347) | 14 |
| Global mean | 15 | 0.763 | 0.976 | 0.214 (0.025–0.389) | 15 |
| LLM-Check | 15 | 0.660 | 0.931 | 0.271 (0.045–0.450) | 15 |
| ICR Probe | 7 | 0.765 | 0.945 | 0.180 (0.027–0.321) | 6 |

Values are means over combinations.

- Over all 67 detector–combination pairs, AUROC_fresh averages 0.959 and AUROC_rec averages 0.737.
- The gap's 95% interval excludes zero in 65 pairs. The two exceptions are DS-L8 Factuality, which has only 39 recovered positions.
- Learned probes, a training-free score and a dedicated streaming detector show the same pattern.

### 2. The deficit is shared across generators and domains

| Generator | Mean AUROC_rec | Mean gap |
|---|---:|---:|
| DS-L8 | 0.829 | 0.128 |
| Gemma-4-31B-it | 0.642 | 0.308 |
| Qwen3.8-27B | 0.733 | 0.235 |

| Domain | Mean AUROC_rec | Mean gap |
|---|---:|---:|
| Math | 0.712 | 0.264 |
| Science | 0.727 | 0.245 |
| Logic | 0.716 | 0.255 |
| Multi-hop | 0.721 | 0.212 |
| Factuality | 0.806 | 0.140 |

## Interpretation

Existing Prefix detectors behave like detectors of "an error has occurred in the trajectory". They do not behave like estimators of κ, the probability that an earlier error is still active. Their scores stay high after an error is corrected.

Pooling past states, as global mean and Streaming do, can record that an error happened. It cannot show that the error was later withdrawn. A κ estimator needs evidence of correction: the current Step must be compared against the earlier content it may still rely on.

AUROC_rec is the direct test of that ability.

## Limitations

- Recovered clean positions come from a limited number of VAL trajectories per combination: 26–177, and 70 for Qwen Logic. Intervals are wide where this number is small.
- Fresh clean positions lie mostly early in trajectories, so AUROC_fresh may also reflect position. This does not affect the low AUROC_rec.

## Artifacts

- [Configuration](config.json): definitions, scope and integrity checks.
- [Recovery JSON](result/recovery.json): unrounded statistics, confidence intervals, position counts, pending cells and source-file hashes.
- [Recovery CSV](result/recovery.csv): one row per detector, generator and domain.

Only aggregate statistics and input-file identities are published. Labels, splits, predictions and detector weights are unchanged and not included.
