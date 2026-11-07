# Code

Run commands from the repository root after following [environment setup](../environment/README.md).

## Configure local assets

```bash
cp configs/paths.example.json configs/paths.local.json
# Fill paths.local.json with your dataset, local model, cache and output paths.
python code/tools/prepare_data.py --paths configs/paths.local.json --check-archived-assignment
```

The dataset validator reads the agreed generation/annotation layout without changing it. The archived-assignment check is required for reproducing the published checkpoint results; omit that flag only for an intentionally different dataset split.

## Streaming reproduction

Prepare the native-input sidecars described in [generation](generation/README.md), then:

```bash
python code/tools/extract_features.py --paths configs/paths.local.json \
  --model deepseek_r1_distill_llama_8b --domain Math
python code/baselines/streaming/run.py --paths configs/paths.local.json \
  --model deepseek_r1_distill_llama_8b --domain Math --mode evaluate
```

Use `--mode train` to fit independent layer candidates on TRAIN, select using VAL and evaluate the selected detector on TEST. Output directories must be new. Reuse the validated feature cache when changing detector methods; do not repeat backbone extraction for already available inputs/layers.

## Methods and utilities

Some baselines require separately obtained upstream dependencies. Follow [THIRD_PARTY.md](../THIRD_PARTY.md) before running HaMI, ARS, RACE or ReDe.

- [baselines](baselines/README.md): method recipes, training and feature inputs.
- [ours](ours/README.md): learnable Step and Prefix implementations.
- [generation](generation/README.md): native formatting, hashes, token counts and hidden states.
- `tools/`: dataset validation and feature preparation.
- `tests/`: CPU equation, alignment and integration checks.

```bash
python -m unittest discover -s code/tests -v
```

GPU entry points must run within a GPU allocation. Local paths and generated outputs are excluded from Git. CPU tests do not certify native backbone inference on every supported GPU/runtime.
