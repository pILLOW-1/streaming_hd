# Original A2 on the direct random 6:2:2 split

Compare prediction-weighted error history with the random-split baselines. A2 jointly predicts Step and Prefix states with the original three-class objective. This exploratory split was previously inspected; these results do not establish a causal effect of correction.

The 30 A2 level cells are separate from the 225 baseline cells. Two previously accepted Qwen model/domain fits are reused by exact split, label, selected-checkpoint and saved-prediction identities; at most 13 new model/domain fits are required. Pending cells are blank.

## Configuration

| Setting | Value |
|---|---|
| Split | Whole question-family random 6:2:2, seed 42 |
| Features | Fixed exponential Step/Prefix pools; raw FP32, no L2 |
| History | All earlier original Steps, including unsupervised Steps; exclusive, trajectory-local |
| Objective | Original C18 three-class CE; predicted history weights detached |
| Optimizer | Adam, lr 0.001, weight decay 0.0001 |
| Budget | 100 epochs, patience 10, batch 256 |
| Selection | Joint mean Step/Prefix VAL AUROC, then mean AP, then lower mean marginal BCE |
| Evaluation | Separate VAL F1max thresholds; selected TEST once |

## Results

AUROC (%). Each model/domain has an independent detector.

### Step

**deepseek_r1_distill_llama_8b**

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 |  |  |  |  |  |

**gemma_4_31b_it**

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 |  |  |  |  |  |

**qwen3_8_27b**

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 |  |  | 88.90 | 93.94 | 90.24 |

### Prefix

**deepseek_r1_distill_llama_8b**

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 |  |  |  |  |  |

**gemma_4_31b_it**

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 |  |  |  |  |  |

**qwen3_8_27b**

| Configuration | Math | Science | Logic | Multi-hop | Factuality |
|---|---:|---:|---:|---:|---:|
| Original A2 |  |  | 88.69 | 91.83 | 91.48 |

The two historical Qwen fits retain their original thresholds. Full selected VAL prediction arrays were not included in the Cetus import; saved TEST scores and complete candidate selection records are verified without rerunning either model. New fits retain complete VAL/TEST arrays privately.
