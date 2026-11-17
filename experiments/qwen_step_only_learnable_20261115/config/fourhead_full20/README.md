# Four-head attention + gated residual, full 20 epochs

Compare with the patience-5 [Four-head attention + gated residual](../fourhead_gated/README.md) on Qwen Logic and Factuality. Reuse the same encoder, frozen inputs, labels, random batches of 128 Steps, Adam (learning rate 0.001, weight decay 0.0005), scheduler, and seed 42. Start from seed 42 because the original completed runs retained best weights alone.

Train layers 1–64 for exactly 20 epochs and save per-epoch VAL Step AUROC, AP, and BCE. Compare the best VAL checkpoint with the patience-5 selection on the same curves. Freeze the original VAL selection rule's winner before a single TEST evaluation per domain. Compare TEST AUROC and AP with the published patience-5 result. The two RunPod GPUs process layers 1–32 and 33–64, with Logic first. Pending results remain blank.
