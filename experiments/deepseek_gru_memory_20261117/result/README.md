# Complete DeepSeek results

Rates are percentages; Brier is an unscaled mean squared probability error. Mean ± sample SD over five seeds. Best means use full stored precision; blank domains have no completed supplied results. Machine-readable values and per-seed evidence are in the four domain/configuration JSON files listed in [index.json](index.json).

## Step

### AUPRC (average precision)

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 38.02 ± 0.00 | 31.33 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | **39.13 ± 0.86** | **39.98 ± 1.44** |  |  |  |

### AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 83.05 ± 0.00 | 89.48 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | **83.33 ± 0.19** | **90.52 ± 0.44** |  |  |  |

### Brier

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 0.07157353 ± 0.00000017 † | **0.03484084 ± 0.00000014** |  |  |  |
| GRU + memory (small_joint_w05) | **0.07054674 ± 0.00059594** | 0.03679820 ± 0.00334139 |  |  |  |

### F1

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 8.06 ± 0.02 | 16.56 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | **25.12 ± 2.81** | **38.45 ± 2.43** |  |  |  |

### accuracy

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 90.83 ± 0.00 | **95.77 ± 0.00** |  |  |  |
| GRU + memory (small_joint_w05) | **91.03 ± 0.11** | 95.11 ± 0.51 |  |  |  |

### precision

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | **75.55 ± 0.04** | **55.43 ± 0.00** |  |  |  |
| GRU + memory (small_joint_w05) | 60.05 ± 3.99 | 43.46 ± 6.46 |  |  |  |

### recall

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 4.26 ± 0.01 | 9.73 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | **16.01 ± 2.50** | **35.65 ± 6.33** |  |  |  |

### specificity

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | **99.86 ± 0.00** | **99.65 ± 0.00** |  |  |  |
| GRU + memory (small_joint_w05) | 98.86 ± 0.35 | 97.79 ± 0.79 |  |  |  |

### TP

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 253.40 ± 0.55 | 51.00 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | 953.40 ± 148.60 | 186.80 ± 33.15 |  |  |  |

### TN

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 57008.00 ± 0.00 | 11572.00 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | 56438.00 ± 201.05 | 11356.60 ± 91.93 |  |  |  |

### FP

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 82.00 ± 0.00 | 41.00 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | 652.00 ± 201.05 | 256.40 ± 91.93 |  |  |  |

### FN

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 5701.60 ± 0.55 | 473.00 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | 5001.60 ± 148.60 | 337.20 ± 33.15 |  |  |  |

† Math Streaming Brier is supplied rounded output; raw predictions were unavailable. See [index.json](index.json).

## Prefix

### AUPRC (average precision)

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 91.13 ± 0.00 | 72.95 ± 0.06 |  |  |  |
| GRU + memory (small_joint_w05) | **92.51 ± 0.23** | **79.22 ± 0.87** |  |  |  |

### AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 82.41 ± 0.01 | 87.12 ± 0.02 |  |  |  |
| GRU + memory (small_joint_w05) | **84.53 ± 0.38** | **89.24 ± 0.43** |  |  |  |

### Brier

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 0.14350920 ± 0.00010732 † | 0.13882470 ± 0.00034480 |  |  |  |
| GRU + memory (small_joint_w05) | **0.14323297 ± 0.01013139** | **0.13657272 ± 0.00845536** |  |  |  |

### F1

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | **86.17 ± 0.04** | **70.28 ± 0.08** |  |  |  |
| GRU + memory (small_joint_w05) | 85.90 ± 2.01 | 70.08 ± 3.08 |  |  |  |

### accuracy

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 79.36 ± 0.04 | 79.84 ± 0.14 |  |  |  |
| GRU + memory (small_joint_w05) | **79.48 ± 1.72** | **81.25 ± 0.77** |  |  |  |

### precision

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 82.54 ± 0.06 | 66.81 ± 0.34 |  |  |  |
| GRU + memory (small_joint_w05) | **84.04 ± 2.50** | **72.41 ± 4.88** |  |  |  |

### recall

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | **90.12 ± 0.17** | **74.14 ± 0.25** |  |  |  |
| GRU + memory (small_joint_w05) | 88.22 ± 6.32 | 68.98 ± 9.35 |  |  |  |

### specificity

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 52.57 ± 0.28 | 82.54 ± 0.33 |  |  |  |
| GRU + memory (small_joint_w05) | **57.71 ± 9.79** | **87.07 ± 4.55** |  |  |  |

### TP

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 40528.60 ± 74.43 | 2893.60 ± 9.84 |  |  |  |
| GRU + memory (small_joint_w05) | 39676.40 ± 2841.27 | 2692.20 ± 364.87 |  |  |  |

### TN

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 9501.80 ± 51.47 | 6796.40 ± 27.01 |  |  |  |
| GRU + memory (small_joint_w05) | 10430.20 ± 1769.84 | 7169.40 ± 375.00 |  |  |  |

### FP

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 8571.20 ± 51.47 | 1437.60 ± 27.01 |  |  |  |
| GRU + memory (small_joint_w05) | 7642.80 ± 1769.84 | 1064.60 ± 375.00 |  |  |  |

### FN

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 4443.40 ± 74.43 | 1009.40 ± 9.84 |  |  |  |
| GRU + memory (small_joint_w05) | 5295.60 ± 2841.27 | 1210.80 ± 364.87 |  |  |  |

† Math Streaming Brier is supplied rounded output; raw predictions were unavailable. See [index.json](index.json).

## Recovery

### recovery_boundary_accuracy

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 23.77 ± 0.32 | 36.93 ± 0.97 |  |  |  |
| GRU + memory (small_joint_w05) | 28.27 ± 12.23 | 48.01 ± 10.58 |  |  |  |

### recovery_transition_rate_given_previous_prediction_1

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 0.80 ± 0.03 | 1.09 ± 0.22 |  |  |  |
| GRU + memory (small_joint_w05) | 3.10 ± 1.73 | 11.50 ± 4.33 |  |  |  |

### recovery_detected_in_observed_episode_rate

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 24.34 ± 0.29 | 37.95 ± 0.86 |  |  |  |
| GRU + memory (small_joint_w05) | 29.91 ± 12.22 | 52.83 ± 11.08 |  |  |  |

### post_recovery_prefix_false_positive_rate

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 63.99 ± 0.33 | 48.49 ± 0.81 |  |  |  |
| GRU + memory (small_joint_w05) | 57.76 ± 12.70 | 34.36 ± 11.21 |  |  |  |

### false_recovery_fraction

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 55.27 ± 2.11 | 25.74 ± 2.82 |  |  |  |
| GRU + memory (small_joint_w05) | 55.09 ± 5.24 | 43.07 ± 11.39 |  |  |  |

### recovery_delay_mean_detected_only

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 0.08 ± 0.01 | 0.12 ± 0.02 |  |  |  |
| GRU + memory (small_joint_w05) | 0.13 ± 0.05 | 0.27 ± 0.18 |  |  |  |

Delay excludes undetected episodes: compare it alongside detection rates and the per-seed failure/censoring counts in each collector summary. Conditional switch rates have model-dependent denominators. These descriptive paired changes are not significance tests.

## Definitions and interpretation

Step metrics score the current reasoning step’s gold binary label. Prefix metrics score the supplied gold state of the reasoning prefix at that step. A prefix may recover from 1 to 0; the prefix target is not forcibly converted to a cumulative OR of all previous errors. Positive class 1 denotes the error/hallucination label. Predictions use **probability > 0.5**, so exactly 0.5 predicts 0.

Each seed pools all retained positions in its domain TEST split. Means and sample SDs (denominator n−1) are then computed across the five fixed seeds. Displayed SDs of 0.00 can reflect rounding; the CSV retains full supplied precision. A long trajectory contributes more positions to pooled scores; these are not question-macro scores. The code exports positions and original step IDs so recovery can distinguish adjacent observations from gaps.

| Metric | Definition and interpretation |
| --- | --- |
| AUROC | Ranking discrimination across thresholds; the probability that a random positive outranks a random negative, with ties handled by the standard scorer. Higher is better. |
| AUPRC / AP | Average precision from the precision–recall curve. This export uses average precision, not trapezoidal PR area. Especially useful when positives are rare; compare with positive prevalence and only across compatible cohorts. Higher is better. |
| Precision | TP / (TP + FP): how many positive alerts are correct. |
| Recall | TP / (TP + FN): how many gold-positive positions are detected. |
| F1 | Harmonic mean of precision and recall, evaluated at the fixed threshold. Higher is better. |
| Specificity | TN / (TN + FP): the fraction of gold-negative positions correctly predicted negative. |
| Accuracy | (TP + TN) / N. Can look high on an imbalanced set even with low positive recall. |
| Brier | Mean of (probability − gold label)². Lower is better; reflects both calibration and discrimination, not calibration alone. |
| TP / FP / TN / FN | Counts of correctly positive / falsely positive / correctly negative / missed positive positions. These are counts, not percentages. |

AUROC and AUPRC describe ranking; F1/precision/recall describe one operating point. Report both ranking measures for Step and Prefix. Step positives are sparse here, so AUPRC and recall reveal behavior that AUROC or accuracy alone can hide. Brier adds probability-quality information. No universal metric alone decides which detector is better.

### Recovery definitions

A timed recovery boundary is a gold Prefix 1→0 change at consecutive original steps. Gapped 1→0 observations are reported separately. A zero episode is the observed following run of gold-negative prefixes.

| Metric | Definition and caveat |
| --- | --- |
| Boundary accuracy | Fraction of gold recovery boundaries predicted 0 immediately. Includes a detector already predicting 0 before the boundary. |
| Conditional recovery switch | Fraction predicted to switch 1→0 among boundaries whose previous detector prediction was 1. The denominator depends on the detector. |
| Predicted recovery transition rate | Fraction of all gold recovery boundaries where the detector switches 1→0. |
| Detected in observed zero episode | Fraction of recovery episodes in which a negative prediction is observed during the eligible zero episode. |
| Post-recovery prefix false-positive rate | Fraction of eligible gold-negative positions after recovery still predicted positive. Lower is better. |
| Delay, detected only | Steps from gold recovery to first detected negative prediction, restricted to episodes with a detection. Undetected episodes are excluded, so a low delay alone does not imply better recovery. |
| False recovery fraction | Fraction of adjacent predicted 1→0 transitions whose current gold label is still 1, divided by all adjacent predicted 1→0 transitions. Lower is better; see source code and per-seed denominators. |
| Fixed-window detected rate | Detection by 0, 1, 2, 5 or 10 steps among episodes with known outcome for that window. Report excluded/censored counts alongside it. |

Nulls represent undefined ratios in the collected comparisons. Check denominators before interpreting sparse cells. The recovery implementation is in `../code/gru_memory/recovery.py` (event calculations); the result JSON retains the original collector denominators. Paired differences are GRU + memory minus Streaming for the same seed; positive favors GRU only for higher-is-better scores. The repository’s within-trajectory recovery AUROC is a different metric.
