# Qwen A2 with detached historical Step representations

Compare three learnable Step readouts on Logic and Factuality with gradients stopped through both historical factors.

$$
g_t = \sum_{i<t} \mathrm{sg}(q_i) \mathrm{sg}(e_i), \qquad z_t=b+\theta^\top c_t+w^\top g_t .
$$

The forward history is identical to the matched A2 experiment. The current-step detector still trains the Step readout. The Prefix loss trains the history coefficient and Prefix head.

| Configuration | Token weights | Pooled values |
|---|---|---|
| Learnable aggregation | Original content-only scorer with within-step softmax | Frozen token features |
| Learnable representation | Original fixed exponential weights | Linear-256, BatchNorm, and ReLU |
| Learnable aggregation and representation | Linear score on the shared representation with within-step softmax | The same shared representation |

| Training parameter | Setting |
|---|---|
| Scope | Two domains, three configurations, native layers 1–64 |
| Shards | Layers 1–32 and 33–64 for each configuration and domain |
| Training | C18 three-state CE, Adam, learning rate 0.001, weight decay 0.0001, seed 42, up to 100 epochs, patience 10 |
| Batching | Shuffle whole trajectories. Target up to 256 labeled positions and 24,000 represented tokens per batch. Keep a trajectory intact when it exceeds a batch target. |
| Source | Existing verified Qwen BF16 features and frozen history index |
| A2 path | Use the learned Step representation for both heads' Step and history inputs. Keep fixed Prefix pooling. Detach both the historical Step probability and historical Step representation. |

Select jointly on all 64 VAL layers using mean Step and Prefix AUROC, AP, and then lower BCE. Evaluate the frozen winner once on TEST.

| Step TEST AUROC | Logic | Factuality |
|---|---:|---:|
| Learnable aggregation | | |
| Learnable representation | | |
| Learnable aggregation and representation | | |

| Prefix TEST AUROC | Logic | Factuality |
|---|---:|---:|
| Learnable aggregation | | |
| Learnable representation | | |
| Learnable aggregation and representation | | |

Status: staged for the existing RunPod workers. Each shard starts after its matching parent shard releases the worker lock. TEST results remain pending.

The launcher reuses the existing frozen history index and the matched A2 support modules on the RunPod volume. The changed detector, evaluator, gradient check and shard entry point are versioned here.
