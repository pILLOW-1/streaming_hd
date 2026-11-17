# Historical Streaming comparator

These are the completed Streaming fits from the same DeepSeek frozen-feature and HF split experiment. The local Step probe and Prefix probe use the original pooling and Streaming objective. The recorded recipe uses LBFGS initialization (maximum 100 iterations, L2 1e-4), followed by 20 Prefix fine-tuning epochs with AdamW, learning rate 1e-3, weight decay 0.01, batch size 64, lambda_final=30 and lambda_sync=120. Selection is the fixed final epoch. The decision rule is strict probability > 0.5.

Five seeds: 42, 123, 456, 789, 2026. Layer 16, width 4096. Exact identities are in `../protocol.json`. The compact folder supplies the new GRU+memory implementation and the completed comparator results. The full original Streaming pipeline remains in the separate reproduction archive; it is not duplicated here. The repository's newer own-VAL layer/threshold selection is a different protocol and should not be substituted when claiming to reproduce these numbers.
