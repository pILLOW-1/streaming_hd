# RACE

RACE uses the complete upstream four-component scorer and same-generator auxiliary responses. Required model assets and pinned upstream provenance are documented in the baseline setup guide.

## Optional attention-stack memory reduction

The upstream scorer stacks all native attention layers before averaging over layers and heads. This temporary stack can exceed GPU memory even when the model forward succeeds. The optional helper processes query slices while retaining the same layer-then-head mean, original FP32 attention tensors, inputs and score equations.

```python
from baselines.race.attention_memory import install_chunked_attention_mean

install_chunked_attention_mean(scorer)
```

Call this explicitly after constructing the upstream scorer. It is not enabled automatically. The helper checks the expected upstream function structure and fails if it differs. It does not truncate inputs, alter precision, regenerate auxiliary responses or substitute missing component scores.

CPU validation compared multiple chunk sizes and the original method on a tiny native FP32 Llama: entropy outputs matched exactly and attention means agreed within the declared numerical tolerance. At runtime, first/last query rows are checked against a dense reduction of the same computed native attention tensors, without an additional model forward. This optimization removes the duplicate stack; it does not remove the memory required by the original attention outputs themselves.
