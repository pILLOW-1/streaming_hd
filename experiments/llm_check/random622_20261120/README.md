# LLM-Check on the random 6:2:2 split

Evaluate the original attention statistic on Prefix labels using the frozen unconstrained question-family assignment. This is an exploratory re-split of historically inspected data.

| Configuration | Statistic | Training |
| --- | --- | --- |
| LLM-Check | Original native attention diagonal statistic, official score sign -1 | No detector training |

| Evaluation parameter | Setting |
| --- | --- |
| Eligible layers | All original full-attention blocks; hybrid linear-attention blocks excluded |
| Layer selection | Complete own VAL AUROC, then AP |
| Threshold | Own VAL F1max; strict score > threshold |
| TEST | Selected layer once; independently checked saved scores |

[Accepted results](result/accepted_cells.json) contain full-precision primary and secondary metrics, eligible-layer counts, frozen thresholds and verification hashes. Original raw scalar caches are reused only after canonical input, producer, row and byte-hash checks; missing native statistics are calculated as needed. No raw data, token inputs, attention banks or saved prediction arrays are included.
