# Shared training and evaluation

Use the [original A2 training and selection protocol](../../readout/configurations/training.md). The loss variant adds the lambda candidates defined in [the design](post_error_ranking.md), yielding 192 independent lambda/layer candidates per domain. Retain separate own-VAL thresholds and one frozen selected TEST; verify saved predictions independently on CPU without model replay.

The frozen runtime configuration remains unchanged. Publication of aggregate results was separately authorized on 2026-11-25.
