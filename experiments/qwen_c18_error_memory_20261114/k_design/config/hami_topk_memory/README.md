# HaMI Step and A2 error-memory content

Test whether Top-k token evidence improves A2 memory over the complete Step exponential pool. H1 and H2 share the same token-scoring architecture; each is trained independently.

| Configuration | Step probability | Memory content |
|---|---|---|
| H1 | Mean Top-k sigmoid token scores | Original Step exponential pool |
| H2 | Mean Top-k sigmoid token scores | Mean raw frozen hidden features of those Top-k tokens |

Use k=floor(0.1n)+1 with stable ties, without softmax. Both use the raw original Streaming Prefix pool. History sums earlier original steps, including unsupervised steps, weighted by detached predicted Step probabilities. Refresh historical scores and token indices in eval/no-grad mode after every update; supervised scoring retains training-mode BatchNorm and Dropout. H2 uses5120-dimensional backbone features, not the scorer intermediate.

The conditional logit is b + theta^T c_t + w^T g_t, and the final Prefix probability is q_t + (1-q_t) kappa_t.

| Training parameter | Setting |
|---|---|
| Token scorer | 5120 ->256, BatchNorm, ReLU, Dropout0.3 ->1, sigmoid |
| Objective | C18 three-class CE; not HaMI MIL loss |
| Optimizer | Adam, learning rate0.001, weight decay0.0001, no scheduler |
| Sampling | 256 supervised positions; batch-local trajectory deduplication; seed42 |
| Budget | Independent layers1-64, max100 epochs, patience10 |
| Selection | Joint mean Step/Prefix VAL AUROC, then mean AP, then lower mean BCE; checkpoint and layer |
| Thresholds | Separate Step/Prefix thresholds selected on VAL |
| TEST | Frozen selected model once; saved predictions independently recomputed on Cetus CPU |

Factuality H1/H2 are accepted, both L50/epoch1. Logic is unfinished and remains blank in the report. [Exact configuration](config.json), [executable source](source/worker_fast.py), [H1 result](../../result/Factuality/H1.json), [H2 result](../../result/Factuality/H2.json). Synthetic CPU/CUDA equivalence checks cover outputs, loss, gradients, BatchNorm, Dropout, ties and causality; they are implementation checks, not extra scientific runs.
