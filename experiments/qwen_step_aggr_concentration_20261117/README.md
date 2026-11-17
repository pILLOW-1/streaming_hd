# Qwen pooling weights within erroneous Steps

This VAL diagnostic includes only Steps with the ground-truth Step error label equal to 1. It compares the frozen legacy learnable aggregation checkpoint in each domain with fixed Streaming on the same Steps. The learned scorer reads the existing selected-layer token features.

![Where fixed and learned pooling put weight inside each erroneous Step](error_step_weight_profile.png)

From Step start (0%) to Step end (100%), each plotted bin shows the mean share of the Step's pooling weight in that positional decile. Each token spans an equal part of its Step. Its weight is split proportionally when it overlaps more than one bin. The ten bin shares sum to 100%, and each Step contributes equally. The shaded region marks the final 20%.

[Per-domain values and coverage](result/index.json)
