# Top-3 equal-weight logit fusion

Select the three single-layer Learnable aggregation detectors with the highest Step VAL AUROC. Freeze each detector and average their logits with weights of 1/3 and zero intercept. Apply sigmoid after fusion. Set the threshold by Step VAL F1 and evaluate the frozen fusion on TEST.

| Domain | Frozen detector layers |
| --- | --- |
| Logic | L35, L34, L42 |
| Factuality | L49, L50, L48 |
| Science | L46, L34, L47 |
| Multi-hop | L40, L49, L34 |
