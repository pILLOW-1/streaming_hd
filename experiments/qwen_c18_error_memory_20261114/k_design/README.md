# Conditional Prefix head design

Maintain changes to the conditional head and error-memory design here. Step-input readout changes are tracked separately in [readout](../readout/README.md).

| Configuration | Conditional Prefix design | Existing verified results |
|---|---|---|
| A1 / original C18 | Linear head on the Prefix exponential pool; no error memory. | [Step and Prefix](../README.md#results), [pooled recovery](../README.md#pooled-test-prefix-recovery-auroc) |
| A2 | Add a linear projection of earlier Step representations weighted by detached predicted error probabilities. | [Step and Prefix](../README.md#results), [pooled recovery](../README.md#pooled-test-prefix-recovery-auroc) |
| A3 | Add a rank-64 bilinear interaction between weighted history and the current Step representation. | [Step and Prefix](../README.md#results); historical within-trajectory recovery remains explicitly labeled there. |

The parent report retains the original records and links. New conditional-head configurations and their accepted results will be maintained in this directory; only independently verified results are added.

## Original A2 domain extension

The same fixed Step/Prefix readouts and C18 objective are reused. Multi-hop selects layer 49, epoch 3 on joint VAL metrics and covers 13,460 TEST positions. [Configuration](config/original_a2_multihop_20261119.json), [acceptance and hashes](result/Multi-hop/A2.json). Science selects layer 46, epoch 1 and covers 46,175 TEST positions. [Science configuration](config/original_a2_science_20261119.json), [acceptance and hashes](result/Science/A2.json). Math remains unfinished.

### Step TEST AUROC

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
|---|---:|---:|---:|---:|---:|
| A2 | **0.8775** | **0.8826** | **0.8218** |  | **0.9128** |

### Prefix TEST AUROC

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
|---|---:|---:|---:|---:|---:|
| A2 | **0.8377** | **0.8367** | **0.7650** |  | **0.8494** |

### Inherited vs. recovered TEST AUROC

| Configuration | Logic | Factuality | Science | Math | Multi-hop |
|---|---:|---:|---:|---:|---:|
| A2 | **0.7204** | **0.7476** | **0.7363** |  | **0.7660** |

Recovery uses all inherited TEST positives and recovered negatives pooled across trajectories, scored by the final Prefix probability. The historical within-trajectory results in the parent report keep their original definition.
