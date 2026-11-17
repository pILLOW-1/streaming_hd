"""The fixed optimized GRU + memory recipe; no test-based selection."""
from copy import deepcopy
import math

SEEDS = (42, 123, 456, 789, 2026)
BASE = dict(state_dim=128, num_heads=4, dropout=0.1, lr=3e-4,
            weight_decay=1e-4, clip_norm=1.0, prefix_loss_weight=0.5,
            epochs=20, patience=5, minimum_improvement=1e-6,
            selection="joint_auc", graph_layers=2,
            recent_neighbors=16, semantic_neighbors=16,
            objective="trajectory_mean_step_BCE_plus_weighted_prefix_BCE",
            updates="one_per_trajectory_full_detector_BPTT",
            edge_supervision="none; predictive graph, not validated provenance")
VARIANTS = {name: dict(BASE, architecture=name) for name in
            ("gru_memory",)}

def config_for(variant, smoke=False):
    if variant not in VARIANTS:
        raise ValueError("Unknown variant: " + str(variant))
    config = deepcopy(VARIANTS[variant])
    if smoke:
        config.update(epochs=2, patience=2)
    return config

def selection_value(scores, selection):
    if selection != "joint_auc":
        raise ValueError("This experiment selects only by validation mean step/prefix AUROC")
    values = [scores[t]["AUC"] for t in ("step", "prefix")]
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Both validation targets need two classes for checkpoint selection")
    return sum(values) / len(values)

def seed_for_slot(slot):
    if not 0 <= slot < len(SEEDS):
        raise ValueError("Slot must be 0..4")
    return SEEDS[slot]
