# Answer correctness is not determined by reasoning correctness

## Motivation

Answer correctness and reasoning correctness are not deterministically tied. An unresolved reasoning error can coexist with a correct final Answer; a clean final reasoning Prefix can coexist with an erroneous Answer. We quantify both patterns before designing a detector that models their relationship.

Here, reasoning correctness is operationalized by the label of the **final retained Prefix**, not by whether every earlier Step was correct. These counts describe label agreement; they do not establish statistical independence or identify the cause of an individual disagreement.

## Analysis settings

| Setting | Value |
|---|---|
| Dataset | Retained Streaming HD, 6:2:2, TEST target p >= 70%, with frozen per-combination relaxations |
| Scope | Three generators, five domains; TRAIN + VAL + TEST |
| Unit | One retained model-specific trajectory |
| Prefix endpoint | Last retained Prefix label, C_T |
| Answer label | Independently assigned Answer label, A |
| Denominator | All trajectories in the corresponding model/domain group |
| Computation | Exact label counts; no detector training or model inference |
| Integrity checks | Complete binary labels, matched Step/Prefix lengths, unique sample IDs across splits, source SHA256, all 15,697 trajectories |

## Overall results

| Final Prefix | Answer clean (A=0) | Answer erroneous (A=1) |
|---|---:|---:|
| Clean (C_T=0) | 7,792 | **1,779** |
| Erroneous (C_T=1) | **2,175** | 3,951 |

- **C_T=1, A=0:** 2,175 / 15,697 (**13.86%**).
- **C_T=0, A=1:** 1,779 / 15,697 (**11.33%**).
- **Either mismatch:** 3,954 / 15,697 (**25.19%**).

Conditional on an erroneous final Prefix, **35.50%** of Answers are clean; conditional on a clean final Prefix, **18.59%** of Answers are erroneous. These conditional percentages have different denominators from the table below.

## Results by generator and domain

Counts and percentages use all trajectories in each generator/domain as the denominator.

| Generator | Domain | Trajectories | C_T=1, A=0 | C_T=0, A=1 |
|---|---|---:|---:|---:|
| DS-L8 | Math | 1,028 | 80 (7.78%) | 39 (3.79%) |
| DS-L8 | Science | 1,059 | 197 (18.60%) | 87 (8.22%) |
| DS-L8 | Logic | 1,000 | 226 (22.60%) | 70 (7.00%) |
| DS-L8 | Multi-hop | 1,148 | 210 (18.29%) | 142 (12.37%) |
| DS-L8 | Factuality | 1,000 | 43 (4.30%) | 235 (23.50%) |
| Gemma-4-31B-it | Math | 1,000 | 112 (11.20%) | 104 (10.40%) |
| Gemma-4-31B-it | Science | 1,047 | 160 (15.28%) | 173 (16.52%) |
| Gemma-4-31B-it | Logic | 1,000 | 77 (7.70%) | 90 (9.00%) |
| Gemma-4-31B-it | Multi-hop | 1,153 | 335 (29.05%) | 83 (7.20%) |
| Gemma-4-31B-it | Factuality | 1,000 | 37 (3.70%) | 168 (16.80%) |
| Qwen3.8-27B | Math | 1,000 | 54 (5.40%) | 153 (15.30%) |
| Qwen3.8-27B | Science | 1,088 | 138 (12.68%) | 178 (16.36%) |
| Qwen3.8-27B | Logic | 1,000 | 46 (4.60%) | 67 (6.70%) |
| Qwen3.8-27B | Multi-hop | 1,159 | 327 (28.21%) | 55 (4.75%) |
| Qwen3.8-27B | Factuality | 1,015 | 133 (13.10%) | 135 (13.30%) |


## Interpretation

The final Prefix label and Answer label differ for one quarter of the retained trajectories. This motivates modeling Answer correctness separately while conditioning on reasoning state. Directly reusing the final Prefix detector score for Answer detection is a baseline approximation, rather than a consequence of label equivalence. The observed mismatch rate is not an AUROC ceiling or a measured detector error rate.

## Artifacts

- [Configuration](config.json): counting definitions and scope.
- [Counts JSON](result/counts.json): unrounded values, the full 2x2 contingency table, each split, and portable source hashes.
- [Counts CSV](result/counts.csv): all 15 combinations, with TRAIN/VAL/TEST and their union.

Only aggregate statistics and input-file identities are published. Original labels, split assignments, raw trajectories and training runs are unchanged.
