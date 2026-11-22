# ARS on the random 6:2:2 split

Independent Answer-level detectors use the original auxiliary representation projection and supervised detector on the frozen unconstrained question-family split, seed 42. This is an exploratory re-split of historically inspected data.

All native candidate layers are trained on TRAIN only and selected using complete own VAL AUROC, AP and lower BCE. The selected checkpoint receives an own-VAL F1-maximizing threshold before its single TEST evaluation. Invalid auxiliary variants follow the previously authorized isolation policy; original questions, annotations and TEST remain unchanged.

Verified native feature bytes and completed candidates are reused across allocated nodes. Placement and bounded worker concurrency do not change numerical precision, optimizer state, sample identity or epoch budgets. Results undergo independent CPU verification of saved scores, labels, coverage, candidate selection and thresholds.

See [configuration](config.json) and [accepted results](result/accepted_cells.json). Raw data, feature banks, predictions and weights are not published.
