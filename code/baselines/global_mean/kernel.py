"""Project Global Mean: full visible generated prefix mean, L2, linear BCE.

Actual surviving source: original_streaming/native_pool_pipeline.py and
fit_native_pool_methods.py in the immutable theory evidence bundle.
Step and Prefix have separate independent label targets. Answer extends the
visible generated span through the answer endpoint; prompt tokens are not pooled.
"""
from baselines.streaming.kernel import LinearProbe, teacher_objective


def global_mean(tokens):
    if len(tokens) == 0:
        raise ValueError("Empty visible generated span")
    return tokens.double().mean(0).float()
