# DeepSeek: GRU + memory

Test whether a recurrent state with explicit memory over completed reasoning Steps improves hallucination detection and recovery relative to Streaming. The backbone is frozen DeepSeek-R1-Distill-Llama-8B; fit each domain independently on the recorded HF split. This folder publishes the compact method code and completed Math/Logic comparison, using the configuration/code/result layout of the Qwen Prefix decomposition experiment.

## Configurations

| Configuration | Design |
| --- | --- |
| [Streaming](config/streaming/README.md) | Historical local-Step and Prefix linear probes, original Streaming objective. |
| [GRU + memory](config/gru_memory/README.md) | Current-Step token attention, 128-dimensional GRU, causal Step-memory retrieval; fixed small_joint_w05 recipe. |

## Training parameters

| Parameter | Streaming | GRU + memory |
| --- | --- | --- |
| Frozen layer / feature width | 16 / 4096 | 16 / 4096 |
| Seeds | 42, 123, 456, 789, 2026 | 42, 123, 456, 789, 2026 |
| Optimizer | LBFGS initialization, then AdamW | AdamW |
| Learning rate / weight decay | 1e-3 / 0.01 | 3e-4 / 1e-4 |
| Maximum epochs | 20 | 20 |
| Update unit | 64 trajectories | One trajectory; full detector BPTT |
| Loss | Original Streaming; λ_final=30, λ_sync=120 | Mean Step BCE + 0.5 × mean Prefix BCE |
| State / heads / dropout | Linear probes | 128 / 4 / 0.1 |
| Gradient norm limit | Original recipe | 1.0 |
| Selection | Fixed final epoch | Mean VAL Step/Prefix AUROC |
| Early stopping | None | Patience 5, improvement > 1e-6 |
| Threshold | p > 0.5 | p > 0.5 |

## Results

Mean ± sample SD over five seeds. Rates below are percentages. Bold is the best supplied mean at full precision; all exact ties are bold. Blank cells are unfinished/unavailable results.

### Step TEST AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 83.05 ± 0.00 | 89.48 ± 0.00 |  |  |  |
| GRU + memory (small_joint_w05) | **83.33 ± 0.19** | **90.52 ± 0.44** |  |  |  |

### Prefix TEST AUROC

| Configuration | Math | Logic | Science | Multi-hop | Factuality |
| --- | --- | --- | --- | --- | --- |
| Streaming | 82.41 ± 0.01 | 87.12 ± 0.02 |  |  |  |
| GRU + memory (small_joint_w05) | **84.53 ± 0.38** | **89.24 ± 0.43** |  |  |  |

[All Step, Prefix and recovery metrics](result/README.md) · [Result index](result/index.json) · [Exact protocol](config/protocol.json)

GRU + memory improves ranking scores and Step F1 in both completed domains. Prefix F1 is slightly lower at the fixed threshold. Logic recovery improves alongside more false recovery and substantial seed variability; Math recovery is mixed. These are descriptive comparisons, not significance claims. This fixed-layer/fixed-threshold experiment is separate from the repository's current VAL-selected protocol and from its Qwen κ-composition model.

## Code

`code/gru_memory/model.py` contains the original architecture and a factory restricted to the reported recipe. `train.py`, `metrics.py` and `evaluate.py` inside that package retain the original training, checkpoint selection and evaluation functions. `recovery.py` retains the event-recovery calculations. Small shared runtime helpers preserve resume and artifact checks. The root `train.py` is the command-line entry point. [Source provenance](code/SOURCE.json) distinguishes unchanged numerical code from new packaging.

The feature loader stays in the existing data pipeline. See [code/README.md](code/README.md) for the required adapter and commands. This folder is the GRU method publication, not a copy of the full native feature-extraction or historical Streaming pipeline. Exact end-to-end historical reproduction remains supported by the separate full handoff archive.

Only copy this experiment folder into `streaming_hd/experiments/`. There are no PBS scripts, private path settings, logs, checkpoints, feature tensors or redundant handoff reports. New run outputs belong outside the repository. Completed historical results are unchanged; this compact packaging has not been rerun on GPUs.
