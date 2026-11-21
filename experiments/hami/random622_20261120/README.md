# HaMI ori on the random 6:2:2 split

Independent Step-level detectors for each generator and domain on the frozen unconstrained question-family split (seed 42). This is an exploratory re-split of historically inspected data.

The original ori multiple-instance-learning architecture and optimizer are retained. Candidate layers are trained independently on TRAIN and selected by complete own VAL AUROC, then AP, then lower BCE. On the selected checkpoint, the threshold maximizes complete VAL F1 with strict score > threshold, then precision and largest-threshold ties. Selected TEST is evaluated once; saved predictions, exact IDs, labels, all candidate hashes and CPU metrics are independently verified.

Existing label-independent native token bags are reused by verified identity; only missing source is extracted. The original data, annotations, source precision and Step spans remain unchanged.

[Accepted results](result/accepted_cells.json) provide all metrics, selected layer, threshold, source-assignment and acceptance hashes. Weights, raw data, feature banks and prediction arrays are not uploaded.
