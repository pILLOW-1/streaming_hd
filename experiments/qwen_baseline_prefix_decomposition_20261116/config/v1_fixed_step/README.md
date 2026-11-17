# v1 + fixed-aggregation Step

Start from [v1](../v1/README.md) and substitute the independently VAL-selected legacy Step detector using fixed within-Step weights proportional to exp(3u). Reuse its saved VAL and TEST scores and score TRAIN once from the frozen Step checkpoint. See the [Step-only report](../../../qwen_step_only_learnable_20261115/README.md).
