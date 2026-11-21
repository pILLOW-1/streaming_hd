# ReDe on the unconstrained random 6:2:2 split

Paper-equation reproduction with explicitly approved projection training:20 epochs, projection dimension128, followed by the original100-epoch detector recipe. Each generator/domain uses an independent TRAIN fit and complete own VAL selection across all native candidate layers.

Select by VAL AUROC, then AP and lower BCE. Freeze the selected checkpoint and VAL F1max threshold before selected TEST once. Published results pass independent CPU checks of saved predictions, complete IDs/labels, all VAL candidates, selection, threshold and metrics. Existing compatible raw source inputs are reused; projections and detector fits from other splits are not reused.

See [accepted results](result/accepted_cells.json). No data, features, predictions or weights are published.
