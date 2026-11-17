# SAPLMA decomposition, Step-negative BCE

Freeze the published SAPLMA Step probabilities. Use the endpoint last-token MLP (256–128–64) as a conditional head and train the composed Prefix BCE on Step-negative TRAIN positions. See the [shared protocol](../protocol/README.md).
