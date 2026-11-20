# ICR Probe on the random 6:2:2 split

The original probe receives the complete vector of true native full-attention ICR components. Each generator/domain fit uses the frozen unconstrained question-family split independently. This is an exploratory re-split of historically inspected data.

The original probe architecture, optimization and epoch budget are unchanged. Complete own VAL selects weights by AUROC, AP and lower BCE; its F1-maximizing threshold is frozen before selected TEST once. Saved predictions, input identities, labels, coverage, thresholds and metrics are independently verified on CPU.

The accepted cells reused complete raw label-independent ICR vectors after source/input/byte checks; no backbone forward was needed. Width is 32 for DeepSeek and 60 for Gemma. This is one probe over all components, rather than independent scalar-layer probes.

[Results](result/accepted_cells.json) include full-precision metrics and verification hashes. Data, raw features, prediction arrays and weights are excluded.
