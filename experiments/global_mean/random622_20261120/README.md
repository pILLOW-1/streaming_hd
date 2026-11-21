# Global Mean on the random 6:2:2 split

Evaluate independent Step, Prefix and Answer annotations under the frozen unconstrained question-family assignment. Each generator/domain/level has its own TRAIN fit and complete own VAL selection. This is an exploratory re-split of historically inspected data.

| Setting | Protocol |
| --- | --- |
| Representation | Original visible generated-context mean (prompt excluded), L2 normalized |
| Training | Original observed L-BFGS, lr1, max_iter100, strong Wolfe; normalized linear BCE probe |
| Candidate layers | All32 DeepSeek,60 Gemma or64 Qwen native layers |
| Weight/layer selection | Own complete VAL AUROC, then AP, then lower BCE |
| Threshold | Selected own VAL F1max; strict score > threshold; precision then largest-threshold ties |
| TEST | Frozen selected checkpoint once; independent saved-prediction CPU verification |

Step and Prefix representations stop at the current retained Step; Answer includes the generated Answer. [Configuration](config.json) specifies each representation. Frozen causal boundary corrections are retained. Existing label-independent features are reused by exact input, span, row and byte identity; only missing components are generated. No historical different-split detector fit is reused.

[Accepted results](result/accepted_cells.json) contain full-precision metrics, selected layers, thresholds and verification hashes. Unfinished cells are empty in [the comparison](../../../reports/main_result/comparison_random_split_candidate.md). No data, feature banks, weights or prediction arrays are published.
