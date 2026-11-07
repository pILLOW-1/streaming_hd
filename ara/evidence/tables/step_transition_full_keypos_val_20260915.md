# Full key-space current-token position: independent VAL milestone

Evidence boundary: seed42, three fresh Full fits. These are **VAL** results only; the dependency-gated one-shot TEST jobs were Q and final audit H at this milestone. Historical TEST inspection makes later TEST comparisons exploratory.

| Model | Owner-bound TRAIN PBS ID | Selected epoch | Independently replayed Step VAL AUROC | Step VAL ACC | Step VAL F1 | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---|
| Qwen | 88022.hpc-head01 | 14 | 0.8804003070552713 | 0.8025585797271718 | 0.7120418848167539 | 755ab470893d660ef18e284d805e81d0490320b062e5c75f1a40a7245839d820 |
| Llama | 88024.hpc-head01 | 6 | 0.8753361296545709 | 0.7952194307010216 | 0.7474042957913926 | 4fb0c201b5295219469e69f1dfa4acc283090411fbfb2b1722f31e21108ca195 |
| DeepSeek | 88025.hpc-head01 | 12 | 0.9437125601060604 | 0.9309358398038414 | 0.7289692374981775 | 936a1ace6a97da56410120d48669be0767b7530b8fc6155dd9ac7bde07b5771c |

Grounding: `experiments/streaming_hd/runs/step_transition_full_keypos_token_attention_gru_20260915/audits/VAL_INDEPENDENT_VALIDATION.json` status `"PASS"` at line 2, audit PBS job `"88028.hpc-head01"` at line 4, `"new_fit_count": 3` at line 10 and `"test_accessed": false` at line 11. Qwen selected epoch `"best_epoch": 14` at line 19 and `"auroc": 0.8804003070552713` at line 25; Llama epoch `6` at line 71 and AUROC `0.8753361296545709` at line 77; DeepSeek epoch `12` at line 115 and AUROC `0.9437125601060604` at line 121. Exact checkpoint SHA lines are 42, 94 and 138. The audit file's verified SHA-256 is `a495d6c644ce8010254ccce7d2aeb9cab7a4c67199e54f9fd6e0a7cd0f1cf478`, independently matched by `sha256sum`, and matches `audits/TEST_FREEZE.json`'s `val_validation_sha256`. The audit replayed the complete-trajectory one-update batch logs and epoch equal-trajectory TRAIN means. The exact new production payload hash is `4df859232503f6727a6cb0e06e056cd77ae798c524e4fcc19d025ed6b0f2089b`.

No inference about TEST superiority or attention mechanism is made from this VAL milestone.
