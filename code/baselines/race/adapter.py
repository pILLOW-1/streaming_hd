"""RACE full-component boundary; prohibit upstream's silent dummy fallbacks."""
import importlib.util
import math
import os
from pathlib import Path

from common.upstream import source_directory
UPSTREAM = source_directory("race")
REQUIRED_MODELS = ("embed_tokenizer", "embed_model", "selfcheck_nli", "llm_tokenizer", "llm_model", "nlp", "cot_extract_sys_prompt")


def require_complete_scorer(scorer):
    missing = [key for key in REQUIRED_MODELS if getattr(scorer, key, None) is None]
    if missing:
        raise ValueError(f"RACE_AUXILIARY_ASSETS_MISSING_NO_DUMMY_FALLBACK: {missing}")


def score_components(components):
    names = ("consistency", "cross_uncertainty", "sindex", "entity_consistency")
    if any(name not in components or not math.isfinite(float(components[name])) for name in names):
        raise ValueError("All four actual RACE components required")
    # Exact official calculate_race_score combiner; no invented reweighting.
    return float(components["consistency"] + components["cross_uncertainty"] + components["sindex"]
                 + max(0.0, components["entity_consistency"]))


def calculate(scorer, main, samples):
    require_complete_scorer(scorer)
    if not samples.get("final_answer"):
        raise ValueError("RACE_REQUIRES_SAME_BACKBONE_SAMPLED_RESPONSES; do not substitute other generators")
    result = scorer.calculate_race_score(main, samples)
    if not math.isclose(result["race_score"], score_components(result["components"]), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("RACE_COMPONENT_IDENTITY_MISMATCH")
    return result
