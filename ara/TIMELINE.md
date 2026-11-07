# Research timeline

- Dataset construction: assemble reasoning trajectories from three generators and five domains; retain original Step indices and independent three-level annotations.
- Baseline implementation: preserve official sources and method-specific feature/objective choices; keep incomplete method adaptations explicit.
- Streaming reproduction: native whole-trajectory features, independent model/domain all-layer fits and own-VAL selection.
- Dynamic-prefix analysis: measure trajectories containing one or more Prefix corrections, including at least two 1→0 transitions; compare detector performance on these subsets.
- Split exploration: examine alternative TRAIN/VAL/TEST ratios and TEST transition proportions. Only the selected 6:2:2 p≥70% experiment is published here.
- Answer evaluation: reuse each selected Prefix detector's final score; evaluate independent Answer labels with Answer VAL thresholds.
- 2026-11-07: retain 6:2:2 p≥70%, whose Answer AUROC is below 90% in all 15 model/domain cells. This is a dataset-design choice informed by the examined results, not evidence that a proposed method improves over Streaming.
- 2026-11-07: prepare the portable collaboration repository and preserve selected detector weights. Theory is deferred. Other research implementations are retained with their readiness limitations; their historical results are not imported.

## Open items

- Validate new proposed methods on the retained split.
- Some archived experimental trainers depend on older feature formats or deleted helper sources. They are identified in code/ours/README.md and are not silently substituted with a different method.
- Establish a separate untouched evaluation set if needed for claims after split-design selection.
