# Qwen ordered dual-readout recovery diagnostic

Compare frozen Prefix and conditional Prefix scores on inherited errors versus fresh or recovered clean VAL positions. Replay only the selected detector checkpoints on existing token banks.

| Configuration | Head | Original objective | Domains |
|---|---|---|---|
| C17 | Linear | Marginal BCE | Logic, Factuality |
| C18 | Linear | Three-class CE | Logic, Factuality |
| C19 | MLP | Marginal BCE | Logic, Factuality |
| C20 | MLP | Three-class CE | Logic, Factuality |

| Evaluation setting | Value |
|---|---|
| Checkpoint selection | Original frozen VAL-selected layer and epoch |
| Scoring | One CPU replay on VAL with the original raw FP32 pooled readouts |
| Primary metric | Within-trajectory macro AUROC for inherited versus recovered clean |
| Secondary metrics | Pooled AUROC and pair-weighted within-trajectory AUROC for both comparisons |

Only positions with Y=0 enter the recovery comparisons. A clean position is recovered if any earlier retained position in the same trajectory has C=1. The two scores use identical position IDs and labels. The Step score is used to verify replay against the original VAL metrics. All eight configurations passed the frozen VAL metric check.

Each cell shows pooled AUROC / within-trajectory pair AUROC / equal-weight trajectory macro AUROC. Each score uses the same eligible positions. The CSV records mixed-label trajectory counts, positive and negative counts, and the number of within-trajectory pairs.

## Prefix against recovered clean

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.6977 / 0.5419 / 0.4715 | 0.8107 / 0.6008 / 0.5384 |
| C18 | 0.6909 / 0.5556 / 0.4867 | 0.8084 / 0.6016 / 0.5231 |
| C19 | 0.7000 / 0.5602 / 0.5032 | 0.8054 / 0.6742 / 0.5330 |
| C20 | 0.7061 / 0.5466 / 0.4830 | 0.8263 / 0.6503 / 0.5538 |

## Prefix against fresh clean

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.9830 / 0.9869 / 0.9589 | 0.9751 / 0.9858 / 0.9273 |
| C18 | 0.9807 / 0.9800 / 0.9442 | 0.9757 / 0.9855 / 0.9267 |
| C19 | 0.9846 / 0.9857 / 0.9595 | 0.9715 / 0.9896 / 0.9477 |
| C20 | 0.9842 / 0.9830 / 0.9423 | 0.9759 / 0.9884 / 0.9304 |

## Conditional Prefix (κ) against recovered clean

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.7011 / 0.5353 / 0.4742 | 0.8086 / 0.6028 / 0.5037 |
| C18 | 0.6956 / 0.5407 / 0.4807 | 0.8058 / 0.6052 / 0.5130 |
| C19 | 0.7022 / 0.5507 / 0.4896 | 0.8033 / 0.6693 / 0.5291 |
| C20 | 0.7072 / 0.5368 / 0.4844 | 0.8238 / 0.6377 / 0.5065 |

## Conditional Prefix (κ) against fresh clean

| Configuration | Logic | Factuality |
|---|---:|---:|
| C17 | 0.9852 / 0.9962 / 0.9873 | 0.9784 / 0.9984 / 0.9829 |
| C18 | 0.9865 / 0.9967 / 0.9836 | 0.9788 / 0.9989 / 0.9893 |
| C19 | 0.9866 / 0.9921 / 0.9784 | 0.9755 / 0.9990 / 0.9931 |
| C20 | 0.9879 / 0.9940 / 0.9816 | 0.9779 / 0.9974 / 0.9836 |
