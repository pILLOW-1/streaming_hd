# Learnable Step aggregation with fixed exponential Prefix

Learn token-to-Step aggregation using the same scorer as the [hierarchical arm](hierarchical_learnable.md). Keep the original A2 fixed exponential Prefix representation unchanged, reusing its exact cached values.

The learned Step representation also enters the original error-history memory: sum over strictly earlier original Steps of detached predicted Step-error probability times the learned Step representation. Gradients through the representation remain enabled. Keep the original joint three-class objective; no ranking loss or layer ensemble.

Use the [shared training and selection protocol](training.md). Factuality completed all 64 layers and one selected TEST, followed by independent saved-prediction CPU verification. Other domains have no accepted result here. [Accepted result](../result/Factuality/step_learnable_fixed_prefix.json).
