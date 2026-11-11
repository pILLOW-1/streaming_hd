# Prefix errors are mostly inherited, not new

## Motivation

The manual constraint Step=1 ⇒ Prefix=1 splits every reasoning position into three mutually exclusive outcomes:

| Outcome | Labels | Meaning |
|---|---|---|
| new | Y=1 | the Step introduces a new error |
| inherited | Y=0, C=1 | no new error, but an earlier error is still active |
| clean | C=0 | no unresolved error |

Prefix error probability is therefore Step error probability plus the inherited term, π^C = π^Y + ψ. We measure how large each outcome is before designing a detector. These are label counts; they do not measure how detectable each outcome is from internal signals.

## Analysis settings

| Setting | Value |
|---|---|
| Dataset | Retained Streaming HD, 6:2:2, TEST target p >= 70%, with frozen per-combination relaxations |
| Scope | Three generators, five domains; **TRAIN + VAL only** |
| Unit | One retained Step position |
| Labels | Step Y and Prefix C as annotated; no label is derived or changed |
| Computation | Exact label counts; no detector training or model inference |
| Integrity checks | Binary labels, matched Step/Prefix lengths, zero Y=1/C=0 positions, unique sample IDs, source SHA256 against split manifests, all 12,560 TRAIN/VAL trajectories |

TEST is excluded because these statistics inform method design and because the selected split deliberately enriches TEST with correction transitions.

## Overall results

Over 1,065,087 positions in 12,560 trajectories:

- **new:** 6.93%
- **inherited:** 38.38%
- **clean:** 54.69%

Among Prefix-positive positions, **84.70%** are inherited. Every generator/domain combination lies between 72.26% and 87.67%.

| Domain | Inherited share of Prefix positives |
|---|---:|
| Math | 84.14% |
| Science | 87.32% |
| Logic | 80.48% |
| Multi-hop | 79.33% |
| Factuality | 79.86% |

| Generator | Inherited share of Prefix positives |
|---|---:|
| DS-L8 | 85.67% |
| Gemma-4-31B-it | 82.52% |
| Qwen3.8-27B | 84.00% |

## Results by generator and domain

Percentages of new, inherited and clean use all retained positions in the group; the last column uses Prefix-positive positions.

| Generator | Domain | Trajectories | Positions | new | inherited | clean | Inherited share of Prefix positives |
|---|---|---:|---:|---:|---:|---:|---:|
| DS-L8 | Math | 823 | 175,946 | 9.62% | 58.62% | 31.76% | 85.90% |
| DS-L8 | Science | 847 | 129,574 | 9.49% | 65.86% | 24.65% | 87.40% |
| DS-L8 | Logic | 800 | 35,118 | 7.50% | 27.81% | 64.69% | 78.75% |
| DS-L8 | Multi-hop | 919 | 13,268 | 13.48% | 35.56% | 50.96% | 72.51% |
| DS-L8 | Factuality | 800 | 5,846 | 14.42% | 56.28% | 29.30% | 79.60% |
| Gemma-4-31B-it | Math | 800 | 43,817 | 8.50% | 38.21% | 53.29% | 81.80% |
| Gemma-4-31B-it | Science | 838 | 41,112 | 4.80% | 27.73% | 67.46% | 85.23% |
| Gemma-4-31B-it | Logic | 800 | 24,543 | 1.77% | 8.33% | 89.90% | 82.46% |
| Gemma-4-31B-it | Multi-hop | 923 | 7,336 | 9.76% | 25.42% | 64.82% | 72.26% |
| Gemma-4-31B-it | Factuality | 800 | 11,200 | 4.64% | 24.55% | 70.80% | 84.10% |
| Qwen3.8-27B | Math | 800 | 346,506 | 5.57% | 26.58% | 67.84% | 82.67% |
| Qwen3.8-27B | Science | 871 | 143,581 | 5.01% | 35.61% | 59.38% | 87.67% |
| Qwen3.8-27B | Logic | 800 | 34,598 | 1.53% | 8.78% | 89.69% | 85.11% |
| Qwen3.8-27B | Multi-hop | 927 | 37,935 | 9.05% | 42.73% | 48.22% | 82.53% |
| Qwen3.8-27B | Factuality | 812 | 14,707 | 10.10% | 35.73% | 54.17% | 77.97% |

## Interpretation

Prefix detection is dominated by deciding whether an earlier error is still active, not by detecting a new error at the current Step. The share is stable across generators and domains although total error rates differ widely. It partly reflects how long errors persist under the Prefix definition: an error that is never retracted labels every later position as inherited. Position-weighted totals are dominated by long trajectories; compare combinations with the per-combination table.

## Artifacts

- [Configuration](config.json): counting definitions and scope.
- [Composition JSON](result/composition.json): unrounded counts and percentages for TRAIN, VAL and their union, plus portable source hashes.
- [Composition CSV](result/composition.csv): the same rows in tabular form.

Only aggregate statistics and input-file identities are published. Original labels, split assignments and raw trajectories are unchanged.
