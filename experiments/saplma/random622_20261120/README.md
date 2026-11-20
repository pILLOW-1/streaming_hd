# SAPLMA on the random 6:2:2 split

Evaluate the original SAPLMA readout on independent Answer labels under the frozen question-family random assignment. This is an exploratory re-split of historically inspected data.

| Configuration | Features | Detector |
| --- | --- | --- |
| SAPLMA | Final generated Answer token contextual representation, unnormalized | 256/128/64 ReLU MLP |

| Training parameter | Setting |
| --- | --- |
| Seed | 42 |
| Optimizer | Original Keras-equation Adam, lr .001, beta1 .9, beta2 .999, epsilon_hat 1e-7 |
| Epochs / batch | 5 / 32 |
| Layer and weight selection | Complete own VAL AUROC, then AP, then lower BCE |
| Threshold | Own Answer VAL F1max, strict score > threshold |
| TEST | Frozen selected checkpoint once; independent CPU verification |

[Accepted results](result/accepted_cells.json) retain full-precision metrics, selected layers, frozen thresholds and provenance hashes. Candidate layer counts are model-specific: DeepSeek32, Gemma60 and Qwen64. Valid compact features are reused by exact identity. Method-specific optimizer/RNG/checkpoint directories are independent. A first DS Logic attempt stopped before any SAPLMA update because it addressed a different method checkpoint; its identity gate was retained and the namespace corrected before the accepted fit. No Global Mean fit was repeated. No data, feature banks, weights or saved predictions are included.
