# Experiments

Each subdirectory records one experiment: its method and data configuration, results, and detector checkpoints. Backbone weights, datasets, and feature tensors are stored externally.

| Experiment | Description |
|---|---|
| [streaming_622_p70](streaming_622_p70/README.md) | Streaming at Step, Prefix, and Answer levels on the selected 6:2:2 split, targeting TEST p≥70% with documented cell-specific relaxations. |
| [prefix_answer_endpoint_20261110](prefix_answer_endpoint_20261110/README.md) | Motivation analysis: final reasoning Prefix and Answer correctness are not deterministically tied; exact label contingency counts across three generators and five domains. |

Each experiment distinguishes verified results from unfinished configurations. Missing results remain blank; earlier split explorations are not merged into the retained result set.
