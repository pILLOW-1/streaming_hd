# Research implementations

`models.py` contains self-contained architecture definitions exported from the preserved research sources. `MODEL_BINDINGS.json` records source and exported-definition hashes. No historical experiment directory is needed by the portable entry points.

## Two-stage Step and Prefix

```bash
python code/ours/run.py --paths configs/paths.local.json \
  --model deepseek_r1_distill_llama_8b --domain Math \
  --step-method learnable_aggregation_representation --prefix-method scalar_gate
```

Step choices: `learnable_aggregation`, `learnable_aggregation_representation`, `learned_token`, `self_attention`, `token_relation_residual`, `content_relation`.

Prefix choices: `scalar_gate`, `vector_gate`, `content_relation_gate`, `direct_mlp`.

The portable composition uses Adam for Step training (lr .001, weight decay .0005, batch 128, at most 20 epochs, patience 5), then freezes the selected Step detector. Prefix uses AdamW, whole-trajectory batches and the anchor/synchronization objective. Its global-mean probe initialization is an explicit current runner choice, not a recovered historical helper. All candidate layers and weights are selected on own VAL; TEST is used only after selection. Native token caches are reused without re-extraction.

## Joint Step and Prefix

```bash
python code/ours/joint.py --paths configs/paths.local.json \
  --model deepseek_r1_distill_llama_8b --domain Math --regime j_unified
```

`j_unified` uses AdamW parameter groups; `j_dual` uses separate Step Adam and Prefix AdamW optimizers. Both optimize Step BCE plus Prefix anchor/sync. The Step teacher probability is detached in synchronization, while the Prefix representation path remains differentiable. Selection follows the original joint objective's VAL Prefix AUROC, with the current AP tie rule. Independent Answer labels are evaluated using final Prefix scores.

The method subdirectories preserve historical implementations and validators as research references. Their original launchers target historical data and are not the current entry points. The missing old runtime helpers are replaced in the portable workflows by `common/dataset.py`, `common/token_cache.py`, `models.py`, and the current trainers; no archived scientific result is claimed to have been rerun.

CPU tests cover architecture-source identity, forward/backward execution and fixed-shape causal/reset behavior. These implementations do not add Ours results to the Streaming-only published experiment.
