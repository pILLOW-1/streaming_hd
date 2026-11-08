# RHD

The score equations preserve the pinned upstream reverse-KL depth convention, Step attention aggregation, coefficient of variation, depth mean and delta correlation.

## Single-Step preprocessing

For this benchmark, the researcher approved splitting a single retained Step at the midpoint of its native token span. Two nonempty contiguous halves replace that span **only inside RHD**. Their concatenated token positions exactly equal the original span; prompts, generation text, frozen token IDs, labels and dataset splits are unchanged. Multi-Step trajectories retain their original spans.

This is an explicit benchmark preprocessing amendment, not an upstream default. An audit of all 15 model/domain cells found 257 single-Step trajectories; all had at least two native tokens and were splittable.

Constant-difference Pearson correlation and zero-mean CV remain governed by the original formulas. Undefined values are surfaced explicitly; no zero imputation or silent exclusion is used.
