# SAPLMA on the random 6:2:2 split

Evaluate independent Step, Prefix and Answer annotations under the frozen unconstrained question-family assignment. Each generator/domain/level has its own TRAIN fit and complete own VAL selection. This is an exploratory re-split of historically inspected data.

| Setting | Protocol |
| --- | --- |
| Representation | Original unnormalized native contextual endpoint |
| Training | Original256/128/64 ReLU MLP, five epochs, batch32; Keras-equation Adam lr.001, beta1.9, beta2.999, epsilon_hat1e-7 |
| Candidate layers | All32 DeepSeek,60 Gemma or64 Qwen native layers |
| Weight/layer selection | Own complete VAL AUROC, then AP, then lower BCE |
| Threshold | Selected own VAL F1max; strict score > threshold; precision then largest-threshold ties |
| TEST | Frozen selected checkpoint once; independent saved-prediction CPU verification |

Step and Prefix representations stop at the current retained Step; Answer includes the generated Answer. [Configuration](config.json) specifies each representation. Frozen causal boundary corrections are retained. Existing label-independent features are reused by exact input, span, row and byte identity; only missing components are generated. No historical different-split detector fit is reused.

[Accepted results](result/accepted_cells.json) contain full-precision metrics, selected layers, thresholds and verification hashes. Unfinished cells are empty in [the comparison](../../../reports/main_result/comparison_random_split_candidate.md). No data, feature banks, weights or prediction arrays are published.
