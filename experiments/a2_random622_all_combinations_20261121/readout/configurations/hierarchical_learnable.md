# Hierarchical learnable aggregation

Learn Step token weights, then independently learn weights over the resulting Step representations:

$$e_t=\sum_j \mathrm{softmax}_j(a(h_{t,j}))h_{t,j},\qquad c_t=\sum_{i\leq t}\mathrm{softmax}_{i\leq t}(b(e_i))e_i.$$

Both scorers use Linear(5120,256), BatchNorm, ReLU, Dropout(0.3), Linear(256,1). Pool the raw features, without L2 normalization. Prefix aggregation includes the current Step; error memory includes strictly earlier Steps only.

Use the same q, κ, history and three-class cross-entropy as [the last-token arm](last_token_both.md). Historical q is detached; gradients through learned e remain enabled. [Shared training protocol](training.md). Factuality completed the full search and independently verified saved-prediction evaluation; Logic is paused. [Accepted result](../result/Factuality/hierarchical_learnable.json).
