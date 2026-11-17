# Learnable aggregation, full 20 epochs

Repeat the original Qwen learnable aggregation on Logic and Factuality, keeping the model, features, labels, splits, Adam, random batches of 128 Steps, seed 42 and VAL rank rule. Train all 64 layers for exactly 20 epochs and retain their per-epoch VAL AUROC, AP and BCE. Replay the original patience-5 checkpoint rule on these same VAL histories and compare it with the full-budget winner. Evaluate only the full-budget selected model on TEST and compare it with the previously accepted patience-5 result.
