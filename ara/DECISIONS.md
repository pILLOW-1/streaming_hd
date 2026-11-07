# Decisions

1. Retain the 6:2:2 split targeting TEST p ≥ 70%, using recorded per-cell relaxations and minimum TRAIN/VAL transition representation.
2. Preserve independent Step, Prefix and Answer labels. Prefix state can recover from 1 to 0.
3. Use complete own-domain VAL AUROC to select detector layer and weights; VAL AP breaks ties. Do not use smaller layer IDs or earlier update counts to break exact ties.
4. Use own-level VAL F1max thresholds for ACC/F1. Report AUROC, AP and empirical TPR@1%FPR as primary metrics.
5. Use native whole-trajectory causal extraction, exact model/tokenizer revisions, token IDs and Step spans. Reuse compatible caches; retain corrected causal token-boundary overrides.
6. Streaming Answer scoring takes the final retained Prefix score. Its detector is the frozen Prefix model, evaluated against independent Answer labels.
7. Publish only the retained Streaming experiment initially. Checkpoints contain detector tensors only. Theory release is deferred.
8. Data live on Hugging Face. Repository files carry source references, split IDs, schemas and reproducible entry points.
9. Deletion of local legacy assets requires a separate reviewed target list and explicit approval; repository publication does not authorize deletion.
