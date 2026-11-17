# Streaming + learned score residual

Use within-Step weights proportional to exp(3u + s), where s is the learned residual score. Zero-initialize the scorer output to recover fixed Streaming pooling at initialization. Pool the original frozen token vectors and use the [legacy recipe](../protocol/README.md).
