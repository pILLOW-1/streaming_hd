# Four-group balanced BCE on frozen natural VAL (2026-09-16)

Source audit: `experiments/streaming_hd/runs/step_four_group_balanced_bce_20260916/audits/VAL_FINAL_VALIDATION.json` (`status: PASS`). Production job `88538.hpc-head01` and dependent validation job `88543.hpc-head01` both exited 0. TEST was not accessed.

Every optimizer batch contained 32 examples from each of G00/G01/G10/G11. The corresponding sampled conditional positive rates were exactly 0.5 for both prior-Prefix states. Natural VAL remained in canonical order and was not rebalanced.

| Model | Raw TRAIN G00/G01/G10/G11 | Updates/epoch | Best epoch | Epochs |
|---|---:|---:|---:|---:|
| Qwen | 11948 / 1692 / 4603 / 7599 | 202 | 4 | 9 |
| Llama | 11844 / 1272 / 6552 / 10795 | 238 | 2 | 7 |
| DeepSeek | 30806 / 1093 / 6954 / 5076 | 344 | 1 | 6 |

Candidate minus exact parent on frozen natural VAL:

| Model | Δ Overall AUROC | Δ ACC | Δ F1 | Δ AUC_clean | Δ AUC_error | Δ delta_FP | Δ delta_FN | Δ abs(delta_FP) | Δ abs(delta_FN) | Support gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen | -0.01218326 | -0.00711442 | +0.02120781 | -0.00223763 | -0.01141836 | +0.03330005 | -0.07668698 | +0.03330005 | -0.07668698 | false |
| Llama | -0.01503084 | -0.00893144 | -0.00983533 | +0.00886943 | -0.00598785 | -0.04864345 | -0.13547643 | -0.04864345 | -0.13547643 | false |
| DeepSeek | -0.00624503 | -0.02325668 | -0.03244017 | +0.00278487 | +0.00069351 | +0.01863901 | -0.19931304 | +0.01863901 | -0.17983340 | false |

The predeclared 1-percentage-point materiality sensitivity required at least one conditional-discrimination or absolute-gap improvement and no Overall AUROC/ACC/F1 decline greater than 0.01. None of the three models passed. These are descriptive VAL comparisons and do not establish causal removal of Prefix information.
