# Qwen C18 error-memory readouts

## Exploration tracks

| Track | Scope |
|---|---|
| [Readout exploration](readout/README.md) | Step and memory representations, including Factuality R1/R2. |
| [Conditional head design](k_design/README.md) | Conditional Prefix head and error-memory designs. |

Test whether predicted errors in earlier reasoning steps improve the C18 conditional Prefix readout in Logic and Factuality.

Let $e_t$ and $c_t$ be the Step and Prefix exponential-pooled features, respectively. All three arms use the same readout structure for the Step score $q_t$ and the same rule for the final Prefix score $p_t^C$.

$$
q_t=\sigma(w_Y^\top e_t+b_Y),\qquad
\kappa_t=\sigma(z_t),\qquad
p_t^C=q_t+(1-q_t)\kappa_t.
$$

The base Prefix logit is $b+\theta^\top c_t$. For A2 and A3, the history vector sums the earlier steps in the same trajectory.

$$
g_t=\sum_{i=1}^{t-1}q_i e_i.
$$

Gradients from the history term do not flow through the weights $q_i$.

| Arm | Conditional Prefix logit $z_t$ | Added term |
|---|---|---|
| A1 (original C18) | $b+\theta^\top c_t$ | None |
| A2 (linear history) | $b+\theta^\top c_t+w^\top g_t$ | Linear projection of history |
| A3 (history × current step) | $b+\theta^\top c_t+(Wg_t)^\top(Ve_t)$ | Interaction between history and the current step through two 64-dimensional projections |

| Training parameter | Value |
|---|---|
| Features | Frozen Qwen BF16 token banks, original FP32 exponential readouts |
| Memory | Exclusive prefix sum of earlier step representations weighted by detached current Step probabilities |
| Initialization | A2 added weight zero. A3 W randomly initialized and V zero |
| Training | C18 three-class CE, Adam, seed 42, LR 0.001, weight decay 0.0001, batch 256 |
| Candidates | Layers 1–64, at most 100 epochs, patience 10 |
| Selection | Original joint mean Step/Prefix VAL AUROC, AP, then lower marginal BCE |

Train on TRAIN, select on VAL and evaluate each frozen arm once on TEST. Report overall Step and Prefix AUROC and inherited-versus-recovered within-trajectory AUROC on the matched recovery cohort. Compute memory from all preceding steps and restrict supervised positions to the frozen C18 labels.

Status: independently verified on VAL and TEST.

## Within-trajectory recovery AUROC (historical)

Both recovery metrics use labeled positions where the current step introduces no new error and at least one earlier labeled position had an erroneous prefix. An inherited error is a positive example. A recovered prefix is a negative example.

- **Conditional recovery AUROC** ranks those positions by the conditional head's estimate that an earlier error remains active, given that the current step introduces no error.
- **Prefix recovery AUROC** ranks the same positions by the final Prefix error score, which combines the predicted chance of a new step error with the conditional head's score. The Step prediction can change the ranking even on this subset.

For both metrics, compute AUROC separately within each trajectory that contains both classes, then average those AUROCs with equal weight per trajectory. The same rule applies to VAL and TEST.

## Pooled TEST Prefix recovery AUROC

Positive examples are all inherited TEST positions: the current Step has no new error and the Prefix is erroneous. Negative examples are recovered TEST positions: the current Step has no new error, the Prefix is clean, and an earlier retained Prefix was erroneous. Pool these positions across trajectories and rank them by the final Prefix score; pairs need not belong to the same trajectory.

| Configuration | Logic | Factuality |
|---|---:|---:|
| Streaming | 0.7020 | 0.7205 |
| Best published baseline | 0.7020 (Streaming) | 0.7233 (Global Mean) |
| A1 | 0.7073 | 0.6895 |
| A2 | **0.7204** | **0.7476** |

The best published baseline is defined by overall Prefix TEST AUROC in the existing baseline comparison, not by recovery performance. Logic selects Streaming; Factuality selects Global Mean. The same frozen selected predictions are reused, with no retraining, inference, or new selection.

The matched cohorts contain 1,733 inherited and 6,593 recovered positions in Logic, and 2,025 inherited and 1,408 recovered positions in Factuality. All inherited positives are retained, including 83 Logic and 33 Factuality positions without an earlier observed positive among retained labels. This differs from the historical within-trajectory tables below, which keep their original restricted cohort and averaging rule.

Saved-prediction alignment, labels, source hashes and overall Prefix AUROC were checked. Recovery AUROC agrees between pair-count and ROC-integration calculations. See the [full results and provenance](result/pooled_prefix_recovery_20261119.json), [CSV](result/pooled_prefix_recovery_20261119.csv), and [analysis configuration](config/pooled_prefix_recovery_20261119.json).

## Results

Frozen selections use the original C18 joint VAL rule.

| Configuration | Logic layer / epoch | Factuality layer / epoch |
|---|---:|---:|
| A1 | 42 / 12 | 50 / 5 |
| A2 | 42 / 12 | 47 / 5 |
| A3 | 42 / 5 | 34 / 6 |
| C17 | 35 / 6 | 50 / 5 |

### VAL Step AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.9158 | 0.8922 |
| A2 | **0.9158** | 0.8918 |
| A3 | 0.9120 | 0.8833 |
| C17 | 0.9151 | **0.8931** |

### VAL Prefix AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.8946 | 0.9379 |
| A2 | **0.9062** | **0.9395** |
| A3 | 0.8825 | 0.9368 |
| C17 | 0.8952 | 0.9379 |

### VAL Conditional recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5637 | 0.6017 |
| A2 | 0.5652 | 0.5958 |
| A3 | 0.5589 | **0.6069** |
| C17 | **0.5747** | 0.5922 |

### VAL Prefix recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5448 | 0.5807 |
| A2 | 0.5546 | 0.5967 |
| A3 | 0.5546 | 0.5977 |
| C17 | **0.5550** | **0.5983** |

### TEST Step AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.8775 | 0.8820 |
| A2 | 0.8775 | **0.8826** |
| A3 | 0.8816 | 0.8753 |
| C17 | **0.8927** | 0.8818 |

### TEST Prefix AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.8341 | 0.8106 |
| A2 | **0.8377** | **0.8367** |
| A3 | 0.8129 | 0.8158 |
| C17 | 0.8370 | 0.8133 |

### TEST Conditional recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5470 | 0.5367 |
| A2 | **0.5583** | 0.5361 |
| A3 | 0.5363 | **0.5469** |
| C17 | 0.5417 | 0.5441 |

### TEST Prefix recovery AUROC

| Configuration | Logic | Factuality |
|---|---:|---:|
| A1 | 0.5492 | **0.5338** |
| A2 | **0.5636** | 0.5226 |
| A3 | 0.5360 | 0.5289 |
| C17 | 0.5498 | 0.5323 |

## A2 comparison with published baselines

All scores are overall TEST AUROC (%) for Qwen3.8-27B. The strongest baseline is the highest independently verified baseline in the [main comparison table](https://github.com/pILLOW-1/streaming_hd/blob/635779c0573feab584a635d199d2ac9328ff1581/reports/main_result/comparison.md) for that domain and detection level. Differences are in percentage points.

### Step

| Domain | A2 | Streaming | A2 − Streaming | Strongest baseline | A2 − strongest |
|---|---:|---:|---:|---|---:|
| Logic | 87.75 | 87.16 | +0.59 | HaMI (88.75) | −1.00 |
| Factuality | 88.26 | 87.20 | +1.06 | HaMI (87.35) | +0.91 |

### Prefix

| Domain | A2 | Streaming | A2 − Streaming | Strongest baseline | A2 − strongest |
|---|---:|---:|---:|---|---:|
| Logic | 83.77 | 82.56 | +1.21 | Streaming (82.56) | +1.21 |
| Factuality | 83.67 | 82.22 | +1.45 | Global Mean (82.26) | +1.41 |
