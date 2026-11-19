# Conditional Prefix head design

Maintain changes to the conditional head and error-memory design here. Step-input readout changes are tracked separately in [readout](../readout/README.md).

| Configuration | Conditional Prefix design | Existing verified results |
|---|---|---|
| A1 / original C18 | Linear head on the Prefix exponential pool; no error memory. | [Step and Prefix](../README.md#results), [pooled recovery](../README.md#pooled-test-prefix-recovery-auroc) |
| A2 | Add a linear projection of earlier Step representations weighted by detached predicted error probabilities. | [Step and Prefix](../README.md#results), [pooled recovery](../README.md#pooled-test-prefix-recovery-auroc) |
| A3 | Add a rank-64 bilinear interaction between weighted history and the current Step representation. | [Step and Prefix](../README.md#results); historical within-trajectory recovery remains explicitly labeled there. |

The parent report retains the original records and links. New conditional-head configurations and their accepted results will be maintained in this directory; this reorganization adds no new training or unverified metrics.
