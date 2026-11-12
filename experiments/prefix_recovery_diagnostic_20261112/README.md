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

Each AUROC is further split by where the two positions of a pair come from:

- **between:** the inherited and the clean position belong to different trajectories.
- **within:** both belong to the same trajectory. Within-trajectory fresh clean positions necessarily precede that trajectory's first error.
- **within, trajectory mean:** the AUROC is computed inside each trajectory that contains both classes, then averaged with equal weight per trajectory.

Each AUROC above is the pair-count-weighted mean of its between and within parts.

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

### 3. Within a trajectory, detectors mark the first error but not later corrections

Errors and corrections alternate within trajectories: 90.3% of VAL inherited positions are preceded by at least one correction (Prefix 1→0) in their own trajectory. Whether an error is still active is therefore mainly a within-trajectory question.

Within-trajectory pairs, however, are only 2.0% of inherited–recovered pairs on average (at most 4.7%), so AUROC_rec above mostly measures between-trajectory pairs.

| Detector | fresh between | fresh within | fresh within, traj. mean | rec between | rec within | rec within, traj. mean |
|---|---:|---:|---:|---:|---:|---:|
| Streaming | 0.976 | 0.989 | 0.965 | 0.767 | 0.641 | 0.550 |
| SAPLMA | 0.957 | 0.940 | 0.905 | 0.751 | 0.629 | 0.578 |
| Global mean | 0.976 | 0.994 | 0.978 | 0.766 | 0.634 | 0.526 |
| LLM-Check | 0.931 | 0.978 | 0.984 | 0.661 | 0.619 | 0.499 |
| ICR Probe | 0.944 | 0.972 | 0.894 | 0.766 | 0.648 | 0.508 |

Values are means over combinations (15 per detector, 7 for ICR Probe).

- Inside a trajectory, positions before the first error are separated from later inherited positions: within-trajectory fresh AUROC averages 0.975.
- After the first error, active and corrected positions of the same trajectory are barely separated:
  - within-trajectory recovered AUROC averages 0.633 (range 0.387–0.789);
  - its trajectory mean averages 0.535, close to chance.
- Within-trajectory recovered AUROC is below its between-trajectory value in 60 of 67 detector–combination pairs. The ≈ 0.74 of AUROC_rec therefore comes mainly from differences between trajectories.

## Interpretation

Existing Prefix detectors behave like detectors of "an error has occurred in the trajectory". They do not behave like estimators of κ, the probability that an earlier error is still active. Within a trajectory, their scores rise at the first error but do not follow later corrections and renewed errors. What remains across trajectories reflects differences in their overall level.

This suggests focusing on how to model κ better, so that corrected errors can be distinguished from errors that are still active. The within-trajectory recovered AUROC, especially its trajectory mean, measures this distinction more directly than AUROC_rec, which between-trajectory differences inflate.

## Limitations

- Recovered clean positions come from a limited number of VAL trajectories per combination: 26–177, and 70 for Qwen Logic. Intervals are wide where this number is small.
- Fresh clean positions lie mostly early in trajectories, so AUROC_fresh may also reflect position. Within a trajectory, "after the first error" and "later in the trajectory" cannot be separated by this analysis. This does not affect the low AUROC_rec.
- A near-chance within-trajectory AUROC may partly reflect how consistently Prefix labels are assigned over time within a trajectory. The present data cannot separate this from detector behaviour.

## Artifacts

- [Configuration](config.json): definitions, scope and integrity checks.
- [Recovery JSON](result/recovery.json): unrounded statistics, confidence intervals, position counts, pending cells and source-file hashes.
- [Recovery CSV](result/recovery.csv): one row per detector, generator and domain.
- [Within-trajectory JSON](result/within.json) and [CSV](result/within.csv): between, within and trajectory-mean AUROC for both negative types, with within-trajectory confidence intervals and pair counts.

Only aggregate statistics and input-file identities are published. Labels, splits, predictions and detector weights are unchanged and not included.
