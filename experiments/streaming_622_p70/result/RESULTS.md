# Streaming Results — 6:2:2, TEST p ≥ 70%

TEST p is the proportion of trajectories with at least two Prefix 1→0 transitions. The target is relaxed where infeasible; actual proportions are reported below. All metrics are percentages.

Layers and checkpoints are selected using complete own-domain Prefix/Step VAL AUROC, with VAL AP as the tie-breaker. ACC and F1 use the selected model’s own-level VAL F1-maximizing threshold (strict score > threshold). Answer scores reuse the final retained Prefix score of the frozen Prefix model, evaluated against independent Answer labels; Answer thresholds are selected on Answer VAL. TPR@1%FPR is the maximum empirical TEST TPR at FPR ≤ 1%, without interpolation.

## Step-Level

| Model | Domain | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---:|---:|---:|---:|---:|
| DS | Math | 83.17 | 38.49 | 14.64 | 87.16 | 40.95 |
| DS | Logic | 89.81 | 31.86 | 19.08 | 92.26 | 36.43 |
| DS | Science | 82.96 | 39.03 | 13.72 | 86.52 | 42.46 |
| DS | Multi-hop | 83.27 | 50.79 | 17.61 | 85.80 | 47.89 |
| DS | Factuality | 80.13 | 53.80 | 10.20 | 76.15 | 50.85 |
| Qwen | Math | 78.52 | 20.95 | 9.14 | 88.16 | 28.61 |
| Qwen | Logic | 87.16 | 20.42 | 21.29 | 95.35 | 26.71 |
| Qwen | Science | 81.45 | 21.98 | 10.38 | 90.77 | 28.63 |
| Qwen | Multi-hop | 89.53 | 58.48 | 33.94 | 92.66 | 54.05 |
| Qwen | Factuality | 87.20 | 44.16 | 16.64 | 87.19 | 44.43 |
| Gemma | Math | 80.40 | 27.10 | 9.24 | 88.89 | 32.71 |
| Gemma | Logic | 80.69 | 12.46 | 8.05 | 93.44 | 21.40 |
| Gemma | Science | 79.51 | 19.58 | 9.52 | 90.50 | 26.15 |
| Gemma | Multi-hop | 82.47 | 43.88 | 23.70 | 92.12 | 45.13 |
| Gemma | Factuality | 79.34 | 16.50 | 5.69 | 91.96 | 23.26 |

## Prefix-Level

| Model | Domain | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---:|---:|---:|---:|---:|
| DS | Math | 82.28 | 91.08 | 13.76 | 79.53 | 86.68 |
| DS | Logic | 87.07 | 73.06 | 13.30 | 79.95 | 66.84 |
| DS | Science | 86.28 | 94.68 | 14.36 | 84.84 | 90.90 |
| DS | Multi-hop | 85.95 | 86.48 | 23.16 | 76.58 | 78.75 |
| DS | Factuality | 80.29 | 90.32 | 31.49 | 71.33 | 77.90 |
| Qwen | Math | 73.57 | 56.61 | 7.51 | 64.01 | 58.20 |
| Qwen | Logic | 82.56 | 39.81 | 7.39 | 78.82 | 45.25 |
| Qwen | Science | 76.16 | 66.45 | 3.12 | 66.27 | 68.15 |
| Qwen | Multi-hop | 83.44 | 82.59 | 11.97 | 74.67 | 78.33 |
| Qwen | Factuality | 82.22 | 79.02 | 10.45 | 73.16 | 74.14 |
| Gemma | Math | 80.44 | 75.65 | 10.89 | 70.05 | 72.90 |
| Gemma | Logic | 77.90 | 29.29 | 4.69 | 72.81 | 38.16 |
| Gemma | Science | 80.43 | 63.66 | 8.86 | 68.83 | 62.39 |
| Gemma | Multi-hop | 75.06 | 61.65 | 9.65 | 68.72 | 68.28 |
| Gemma | Factuality | 66.62 | 38.34 | 0.80 | 59.53 | 52.88 |

## Answer-Level

| Model | Domain | AUROC | AP | TPR@1%FPR | ACC | F1 |
|---|---|---:|---:|---:|---:|---:|
| DS | Math | 82.15 | 79.67 | 6.54 | 74.15 | 78.88 |
| DS | Logic | 70.59 | 46.63 | 0.00 | 62.00 | 62.38 |
| DS | Science | 58.94 | 76.78 | 2.10 | 66.98 | 80.23 |
| DS | Multi-hop | 66.08 | 76.76 | 4.76 | 65.50 | 78.71 |
| DS | Factuality | 66.51 | 84.15 | 14.29 | 70.00 | 82.35 |
| Qwen | Math | 85.13 | 66.56 | 3.17 | 81.50 | 72.18 |
| Qwen | Logic | 82.29 | 40.95 | 8.70 | 58.50 | 34.65 |
| Qwen | Science | 57.73 | 39.18 | 2.90 | 51.61 | 46.70 |
| Qwen | Multi-hop | 73.31 | 42.31 | 7.55 | 63.79 | 46.84 |
| Qwen | Factuality | 83.69 | 80.63 | 30.59 | 67.98 | 69.77 |
| Gemma | Math | 77.58 | 62.03 | 3.80 | 69.50 | 70.53 |
| Gemma | Logic | 85.53 | 51.30 | 3.33 | 81.00 | 52.50 |
| Gemma | Science | 57.49 | 36.12 | 0.00 | 45.45 | 44.66 |
| Gemma | Multi-hop | 67.12 | 32.64 | 2.33 | 75.65 | 37.78 |
| Gemma | Factuality | 87.85 | 77.97 | 27.12 | 85.50 | 75.21 |

## TEST Composition

Hallucination rates use all retained positions for Step/Prefix and trajectories for Answer. c is the proportion of all ≥2-transition trajectories assigned to TEST.

| Model | Domain | TEST trajectories | Step hallucination | Prefix hallucination | Answer hallucination | p | c |
|---|---|---:|---:|---:|---:|---:|---:|
| DS | Math | 205 | 9.45% | 71.33% | 52.20% | 89.76% | 26.47% |
| DS | Logic | 200 | 4.32% | 32.16% | 35.50% | 70.00% | 31.89% |
| DS | Science | 212 | 9.88% | 77.43% | 67.45% | 86.32% | 23.64% |
| DS | Multi-hop | 229 | 14.28% | 51.46% | 64.19% | 71.18% | 67.92% |
| DS | Factuality | 200 | 17.27% | 67.92% | 70.00% | 1.50% | 27.27% |
| Qwen | Math | 200 | 6.00% | 32.83% | 31.50% | 81.00% | 26.09% |
| Qwen | Logic | 200 | 2.04% | 13.41% | 11.50% | 70.00% | 43.08% |
| Qwen | Science | 217 | 5.19% | 43.35% | 31.80% | 87.56% | 24.64% |
| Qwen | Multi-hop | 232 | 8.60% | 51.84% | 22.84% | 74.14% | 36.06% |
| Qwen | Factuality | 203 | 9.64% | 46.53% | 41.87% | 70.44% | 59.83% |
| Gemma | Math | 200 | 7.58% | 45.56% | 39.50% | 74.50% | 29.22% |
| Gemma | Logic | 200 | 2.81% | 12.71% | 15.00% | 71.50% | 51.07% |
| Gemma | Science | 209 | 5.02% | 31.85% | 29.67% | 79.43% | 28.82% |
| Gemma | Multi-hop | 230 | 7.20% | 37.11% | 18.70% | 14.35% | 41.25% |
| Gemma | Factuality | 200 | 4.89% | 30.33% | 29.50% | 33.50% | 62.62% |
