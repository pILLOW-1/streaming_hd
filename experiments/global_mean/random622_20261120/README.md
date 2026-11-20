# Global Mean on the random 6:2:2 split

Evaluate the original Global Mean linear readout on independent Answer annotations using the frozen, unconstrained question-family random split. This is an exploratory re-split of historically inspected data.

| Configuration | Features | Detector |
| --- | --- | --- |
| Global Mean | Mean of all visible generated reasoning and Answer token features; prompt excluded; L2 normalization | Linear BCE detector |

| Training parameter | Setting |
| --- | --- |
| Seed / layers | 42 / all 32 DeepSeek or 64 Qwen layers |
| Optimizer | Original observed L-BFGS, lr 1, max_iter 100, strong Wolfe |
| Layer and weight selection | Complete own VAL AUROC, then AP, then lower BCE |
| Decision threshold | Own Answer VAL F1max; strict score > threshold |
| TEST | Frozen selected checkpoint once; independent CPU metrics, IDs, labels and coverage checks |

## Answer TEST AUROC (%)

| Configuration | Math | Science | Logic | Multi-hop | Fact |
| --- | ---: | ---: | ---: | ---: | ---: |
| DS Global Mean | | | **82.71** | | |
| Qwen Global Mean | | | **94.56** | **88.99** | **92.37** |

Full-precision primary and secondary metrics, selected layers, frozen thresholds and verification hashes are in [the result index](result/index.json). Unfinished cells remain empty. Compatible label-independent features were reused by exact identity; missing features were extracted only as needed. No datasets, native feature banks, detector weights or prediction arrays are included.
