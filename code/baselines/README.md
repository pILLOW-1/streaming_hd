# Baselines

All methods consume the agreed local dataset through `common/dataset.py`. No program downloads or replaces the dataset. Each generator/domain is an independent fit. External data, models, feature caches and outputs are configured in the private paths file.

| Method | Level | Implementation / entry | Required features |
|---|---|---|---|
| Streaming | Step, Prefix, Answer | `streaming/run.py` | Native local/global exponential pools; Answer reuses the final Prefix score |
| SAPLMA | Step, Prefix, Answer | `run.py`, `saplma/kernel.py` | Last contextual token vector for the target endpoint |
| Global Mean | Step, Prefix, Answer | `run.py`, `global_mean/kernel.py` | Mean of visible generated tokens, excluding the prompt |
| ICR Probe | Prefix | `run.py`, `icr_probe/native.py` | Native attention and actual incoming/outgoing states for each eligible block |
| HaMI | Step | `run.py`, `hami/kernel.py` | Token bags and same-backbone semantic-cluster probabilities |
| ARS + probe | Answer | `run.py`, `ars/kernel.py` | Original embeddings and TRAIN-only answer-agreement variants |
| LLM Check | Prefix | `run.py`, `llm_check/kernel.py` | Native attention diagonal statistic |
| RACE | Answer | `run.py`, `race/adapter.py` | Four genuine components, auxiliary models and same-backbone samples |
| RHD | Answer | `run.py`, `rhd/kernel.py` | Four components derived from native depth/entropy/Step-attention |
| ReDe + probe | Answer | `rede/run.py`, `rede/pipeline.py` | Full native generated-Answer inputs; projection training and filtered-context re-forward |

## Running a detector

```bash
python code/baselines/run.py --paths configs/paths.local.json \
  --recipe configs/baselines/saplma.step.json \
  --model deepseek_r1_distill_llama_8b --domain Math
```

Recipes retain each method's architecture, loss and optimizer. The current evaluation protocol selects detector weights/layers using complete own VAL AUROC, then VAL AP. Exact unresolved best ties stop explicitly; there is no preference for smaller layers or earlier updates. Freeze the selected detector, then select the VAL F1max threshold and evaluate TEST. No TEST threshold or model selection is allowed. Raw-score methods use explicitly configured score direction/combination weights.

The generic feature location is:

```text
<baseline_feature_root>/<method>/<model>/<domain>/<level>/layer_<index>/
  train.npz  train.json
  val.npz    val.json
  test.npz   test.json
```

Each pack contains ordered `position_id` values (`sample_id:original_step_id`, or `sample_id:answer`) and features only. Its JSON receipt binds the SHA256, method, level, layer and source identities. Labels always come from the dataset, never from a feature pack. Layer zero is a pack identifier for an all-layer/component feature vector, not an embedding-layer selection.

## Feature preparation

- **Streaming:** `code/tools/extract_features.py`; reusable native pools.
- **SAPLMA / Global Mean Step and Prefix:** extract raw token states with `code/generation/generation.py` (omit `--pool-only`), then use `code/tools/pack_hidden_features.py`. Existing token tensors are reused without another model forward. For Answer, use `code/tools/extract_answer_features.py` with full native generated-Answer sidecars; it computes both methods from the same forward. Do not relabel the final reasoning vector as an Answer-token feature.
- **ICR / LLM Check:** `code/tools/extract_attention_features.py`. ICR additionally needs exact native `core_positions` (`user_prompt_start`, `user_prompt_end`, `response_start`) in the feature-input sidecar. Qwen linear-attention blocks are excluded; the remaining blocks retain their original decoder indices.
- **RHD:** `code/tools/extract_rhd_features.py`, with explicit `--anchor-layers` and `--attention-layers`. Layer assignments and component weights for the current generators are adaptations; the released upstream constants were for different models. Undefined upstream statistics on degenerate traces raise an error rather than being silently replaced or dropped.
- **RACE:** `code/tools/extract_race_features.py`, using configured local auxiliary models and sampled-response JSONL. Pinned upstream sampling code is obtained directly from its author as described in `THIRD_PARTY.md`. The wrapper rejects every missing auxiliary model and dummy fallback.
- **HaMI / ARS:** pinned upstream sampling, semantic-clustering, variant and embedding implementations are external dependencies installed using `code/tools/prepare_upstream.py`; see `THIRD_PARTY.md`. `code/tools/assemble_auxiliary_features.py` joins and verifies those assets against this dataset before the shared trainer consumes them. HaMI semantic probabilities must come from sampled semantic clusters. ARS variant agreement must compare answers, not use hallucination labels; variants must remain inside TRAIN.
- **ReDe:** `rede/run.py` performs projection fitting, filtering, native re-forward, probe training and VAL selection. See its README for the full-input sidecar and explicit paper-vs-project choices.

Native-attention producers are dense reference implementations. They require memory for the complete trajectory and never truncate, quantize or fabricate attention to fit an allocation. Their three-backbone GPU execution is not claimed validated by CPU tests.

## Source fidelity and verification

Pinned upstream attribution is retained; redistributable sources are included and other implementations are obtained separately under their author terms. RHD preserves the upstream reverse-KL-to-mixture statistic named “JS”, including its vocabulary mean and scale factor; it does not silently substitute textbook JSD. ReDe equations and the missing projection/filter orchestration are implemented from the paper, with unspecified choices visible in its recipe.

The published result set belongs only to Streaming. Added method code and CPU equation tests are not new benchmark results. Original upstream scripts retain their own input interfaces; use the portable entry points above for this dataset.
