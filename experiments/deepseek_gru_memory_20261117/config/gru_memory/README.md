# GRU + memory — small_joint_w05

Encode the frozen token features of each current reasoning Step with four-head attention, using the same Step's mean as query. A gated residual forms its 128-dimensional representation and local Step logit. Update a normalized GRU state, retrieve completed Step representations through the current Step using four-head attention with a learned null key and exactly zero null value, and form a gated residual Prefix readout. Memory does not enter the local Step encoder. Language-model features themselves retain their causal Prefix context.

Train with mean Step BCE + 0.5 × mean Prefix BCE, one complete trajectory per optimizer update with full detector BPTT. AdamW: learning rate 3e-4, weight decay 1e-4; dropout 0.1; gradient norm limit 1.0. Maximum 20 epochs, patience 5, minimum improvement 1e-6. Choose the checkpoint using mean own-VAL Step/Prefix AUROC. Use fixed strict p > 0.5, without threshold calibration. Fit seeds 42, 123, 456, 789 and 2026 independently for each domain.

The selected variant is `gru_memory`, despite the historical source package name `method4_graph_v1`. Graph configuration fields in the preserved recipe are inactive. No graph mechanism or Qwen κ composition is selected here.
