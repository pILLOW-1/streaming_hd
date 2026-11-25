# Trajectory-equal VAL checkpoint and layer selection

Motivation: long validation trajectories contribute more positions under ordinary selection. Test the contribution of the selection measure without changing A2 training.

Each VAL position receives weight $w_i=1/n_{g(i)}$, where $n_{g(i)}$ is its trajectory's number of supervised VAL positions. Compute weighted AUROC/AP/BCE across all positions; this is not an average of separate within-trajectory AUROCs.

Select checkpoint and layer jointly by mean weighted Step/Prefix AUROC, then mean weighted AP, then lower mean weighted marginal BCE. Both outputs share that selected checkpoint. Use the original output-specific VAL F1max thresholds. TEST AUROC remains the ordinary position-level metric.

Training is fresh original A2: fixed exponential Step/Prefix pools, detached prediction-weighted causal error memory, original three-class CE and position-weighted batches. No trajectory-equal training loss, architecture change or additional TEST replay.

| Domain | Selected layer | Selected epoch | Accepted candidate layers |
|---|---:|---:|---:|
| Logic | 35 | 9 | 64 |
| Factuality | 47 | 4 | 64 |

[Exact configuration](trajectory_equal_val.json) · [Shared training](training.md) · [Results](../README.md)
