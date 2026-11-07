# Configuration

Copy `paths.example.json` to the ignored `paths.local.json` and fill in the external paths. Empty values are intentional: no cluster-specific path or credential is bundled.

- `dataset_root`: authoritative model/domain/split generation and annotation tree.
- `dataset_roots`: optional per-model/domain overrides for separately stored selected cells.
- `model_paths`: local backbone/tokenizer snapshots.
- `native_input_root`: frozen reasoning-input sidecars for feature extraction.
- `native_full_input_root`: full generated-Answer input sidecars for methods that require them.
- `feature_cache_root`: reusable native token/pool cache.
- `baseline_feature_root`: compact method-specific feature packs.
- `output_root`: a new destination for each experiment's logs, detector weights and results.
- `race_assets`: auxiliary models and sampled responses for RACE.

`model_revisions.json` records the source backbone/tokenizer identities. Baseline recipes are under `baselines/`; the archived Streaming recipe and assignment belong to its experiment directory. Consult the method README before changing a scientific parameter.
