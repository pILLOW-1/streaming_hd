# Prefix detectors track whether an error occurred, not whether it is still unresolved

## Motivation

The [notation document](../../theory/STEP_PREFIX_NOTATION.md) writes Prefix error probability as

```math
\pi^{\mathrm{P}}_t=\pi^{\mathrm{S}}_t+(1-\pi^{\mathrm{S}}_t)\,\kappa_t ,
```

where $`\kappa_t`$ is the probability that an earlier error is still unresolved given that step $`t`$ introduces no new error.

The [composition analysis](../step_prefix_composition_20261111/README.md) shows that about 85% of Prefix positives are inherited ($`y^{\mathrm{S}}_t=0,\ y^{\mathrm{P}}_t=1`$). Prefix detection is therefore mostly a matter of $`\kappa_t`$: when the current Step adds no new error, is an earlier error still unresolved?

A detector can approximate this with a weaker cue: whether any error has occurred earlier in the trajectory. Both cues agree before the first error. They disagree only after an error has been resolved. We use those positions to test which cue a detector actually uses.

![Separation of unresolved errors from clean positions, per detector](result/figure.png)

*Figure. AUROC of each detector's Prefix score on TEST. The positive class is always the same: positions with an inherited error ($`y^{\mathrm{S}}_t=0,\ y^{\mathrm{P}}_t=1`$). Circle: against fresh clean positions, which have no error so far in their trajectory. Square: against recovered clean positions, whose earlier error has been resolved. Each value is the mean over generator/domain combinations (15 per detector, 11 for ICR Probe). The dashed line marks chance.*

## Analysis settings

On TEST positions with $`y^{\mathrm{S}}_t=0`$, the positive class is **inherited** ($`y^{\mathrm{P}}_t=1`$). Negatives are split by history:

| Negative | Definition | What separates it from inherited |
|---|---|---|
| fresh clean | $`y^{\mathrm{P}}_t=0`$, no earlier $`y^{\mathrm{P}}_t=1`$ in the trajectory | "an error occurred" suffices |
| recovered clean | $`y^{\mathrm{P}}_t=0`$, some earlier $`y^{\mathrm{P}}_t=1`$ in the trajectory | requires "the error is still unresolved" |

- **AUROC_fresh:** inherited vs fresh clean.
- **AUROC_rec:** inherited vs recovered clean.
- **gap:** AUROC_fresh − AUROC_rec.

A detector that tracks $`\kappa_t`$ itself has a small gap. A detector that tracks only "an error occurred" has a high AUROC_fresh and a low AUROC_rec.

| Setting | Value |
|---|---|
| Dataset | Retained Streaming HD, 6:2:2; **TEST only** |
| Scope | Three generators, five domains |
| Scores | Each detector's TEST Prefix score saved by its original run, from the configuration frozen on VAL; no retraining, selection or inference |
| Detectors | Streaming, SAPLMA, global mean, LLM-Check, ICR Probe |
| Integrity checks | Saved position ids and labels equal the dataset loader's in order. Recomputed TEST Prefix AUROC equals the value recorded by the source run. |
| Uncertainty | 95% percentile intervals from 1,000 trajectory-bootstrap replicates |

Nothing is selected or tuned in this analysis, so it is reported on TEST like the detectors' main results. The earlier VAL version is kept in [`result/val/`](result/val/); its conclusions are the same (see the comparison at the end of the findings).

ICR Probe covers 11 of 15 combinations; its run has not finished for Math and Science of DS-L8 and Qwen3.8-27B. Those cells are listed as pending in `result/recovery.json`.

## Findings

### 1. Every detector separates inherited errors from fresh clean positions far better than from recovered ones

| Detector | Combinations | AUROC_rec | AUROC_fresh | gap (range) | gap CI above 0 |
|---|---:|---:|---:|---:|---:|
| Streaming | 15 | 0.698 | 0.954 | 0.256 (0.035–0.427) | 14 |
| SAPLMA | 15 | 0.688 | 0.930 | 0.242 (0.026–0.370) | 14 |
| Global mean | 15 | 0.688 | 0.955 | 0.267 (0.055–0.515) | 14 |
| LLM-Check | 15 | 0.615 | 0.916 | 0.301 (0.146–0.418) | 15 |
| ICR Probe | 11 | 0.597 | 0.898 | 0.301 (0.045–0.419) | 10 |

Values are means over combinations.

- Over all 71 detector–combination pairs, AUROC_fresh averages 0.933 and AUROC_rec averages 0.661.
- The gap's 95% interval excludes zero in 67 pairs. The four exceptions are all DS-L8 Factuality, which has only 24 recovered positions in 15 trajectories.
- Learned probes, a training-free score and a dedicated streaming detector show the same pattern.

### 2. The deficit is shared across generators and domains

| Generator | Mean AUROC_rec | Mean gap |
|---|---:|---:|
| DS-L8 | 0.725 | 0.191 |
| Gemma-4-31B-it | 0.576 | 0.351 |
| Qwen3.8-27B | 0.689 | 0.267 |

| Domain | Mean AUROC_rec | Mean gap |
|---|---:|---:|
| Math | 0.698 | 0.273 |
| Science | 0.713 | 0.251 |
| Logic | 0.639 | 0.307 |
| Multi-hop | 0.635 | 0.291 |
| Factuality | 0.631 | 0.236 |

### 3. TEST and VAL agree

Means over the same 71 detector–combination pairs on both splits.

| Statistic | VAL | TEST |
|---|---:|---:|
| AUROC_fresh | 0.958 | 0.933 |
| AUROC_rec | 0.732 | 0.661 |
| gap | 0.226 | 0.272 |
| gap CI above 0 | 69 / 71 | 67 / 71 |

The pattern is the same and slightly stronger on TEST. The VAL files in `result/val/` were computed when ICR Probe covered 7 combinations; the VAL column here uses its 11 current combinations.

## Interpretation

Existing Prefix detectors behave like detectors of "an error has occurred in the trajectory". They do not behave like estimators of $`\kappa_t`$, the probability that an earlier error is still unresolved.

This suggests modelling $`\kappa_t`$ explicitly, so that recovered positions can be distinguished from positions whose error is still unresolved. AUROC_rec measures this distinction.

## Limitations

- Recovered clean positions come from a limited number of TEST trajectories per combination: 15–201, and 155 for Qwen Logic. Intervals are wide where this number is small.
- Fresh clean positions lie mostly early in trajectories, so AUROC_fresh may also reflect position. This does not affect the low AUROC_rec.

## Artifacts

- [Configuration](config.json): definitions, scope and integrity checks.
- [Recovery JSON](result/recovery.json): unrounded statistics, confidence intervals, position counts, pending cells and source-file hashes.
- [Recovery CSV](result/recovery.csv): one row per detector, generator and domain.
- [VAL version](result/val/): the recovery JSON and CSV from the earlier VAL analysis.
- Figure: [PDF](result/figure.pdf), [PNG](result/figure.png).

Only aggregate statistics and input-file identities are published. Labels, splits, predictions and detector weights are unchanged and not included.
