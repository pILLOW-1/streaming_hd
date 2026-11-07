"""Accepted causal feature-cache consumption; no implicit feature reconstruction."""
import json
from pathlib import Path

from common.baseline_data import NotReady, checked_asset, file_sha, dataset_samples, canonical_sha


def feature_contract(path, dataset, config, generator):
    path = Path(path)
    if not path.is_file():
        raise NotReady(f"MISSING_ACCEPTED_NATIVE_FEATURE_MANIFEST: {path}")
    value = json.loads(path.read_text())
    expected = {"status": "PASS_CAUSAL_NATIVE_FEATURES", "dataset_manifest_sha256": dataset["_sha256"],
                "label_version": dataset["label_version"], "generator": generator,
                "method_key": config["method_key"], "level": config["level"]}
    for field, item in expected.items():
        if value.get(field) != item:
            raise NotReady(f"FEATURE_IDENTITY_MISMATCH: {field} at {path}")
    for field in ("model_revision", "native_model_config_sha256", "causal_boundary_proof",
                  "tokenizer_sha256", "special_token_policy", "provider_acceptance", "layers"):
        if not value.get(field):
            raise NotReady(f"FEATURE_CONTRACT_FIELD_MISSING: {field}")
    proof = json.loads(checked_asset(value["provider_acceptance"], path.parent).read_text())
    if proof.get("status") != "PASS_NATIVE_FEATURE_SOURCE_AND_CAUSAL_BOUNDARIES":
        raise NotReady("NATIVE_FEATURE_SOURCE_AND_CAUSAL_GATE_PENDING")
    if proof.get("dataset_manifest_sha256") != dataset["_sha256"]:
        raise NotReady("FEATURE_CAUSAL_PROOF_DATASET_MISMATCH")
    if proof.get("feature_recipe_sha256") != canonical_sha(config["feature_recipe"]):
        raise NotReady("NATIVE_FEATURE_RECIPE_NOT_BOUND_TO_THIS_METHOD_AND_LEVEL")
    if proof.get("model_revision") != value["model_revision"]:
        raise NotReady("NATIVE_MODEL_REVISION_NOT_BOUND_TO_FEATURE_PROOF")
    for rule in ("no_future_steps", "no_gold_input", "no_answer_input_for_step_prefix",
                 "exact_token_boundaries", "no_test_label_use", "no_synthetic_standin"):
        if proof.get(rule) is not True:
            raise NotReady(f"FEATURE_CAUSAL_RULE_NOT_PASSED: {rule}")
    if config["method_key"] in {"icr_probe", "llm_check"}:
        if not proof.get("native_attention_mapping_verified"):
            raise NotReady("NATIVE_ATTENTION_MAPPING_PENDING: linear-attention blocks are not invented attention matrices")
    if config["method_key"] == "hami" and proof.get("semantic_probability_source") != "same_backbone_sample_clusters":
        raise NotReady("HAMI_SE_AUGMENTATION_ASSET_MISSING; no labels-as-probabilities substitution")
    if config["method_key"] == "ars":
        if proof.get("variant_supervision_kind") != "answer_agreement_NOT_answer_truth":
            raise NotReady("ARS_TRAIN_ANSWER_AGREEMENT_VARIANTS_MISSING")
    if config["method_key"] == "race":
        if proof.get("all_auxiliary_models_loaded") is not True or proof.get("dummy_fallbacks_used") is not False:
            raise NotReady("RACE_FULL_COMPONENT_AUXILIARY_MODEL_GATE_PENDING")
    value["_path"], value["_sha256"] = str(path.resolve()), file_sha(path)
    return value


def load_pack(features, dataset, config, generator, layer, split):
    import numpy as np
    rows = [r for r in dataset_samples(dataset, config["level"], generator, {split}) if r["eligible"]]
    asset = features["layers"][str(layer)][split]
    path = checked_asset(asset, Path(features["_path"]).parent)
    with np.load(path, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    if any(name in values for name in ("labels", "step_labels", "prefix_labels", "answer_label", "gold_answer")):
        raise NotReady("FEATURES_MUST_NOT_CONTAIN_GOLD_OR_LABEL_COLUMNS")
    ids = list(map(str, values.pop("sample_id")))
    views = list(map(str, values.pop("causal_view_sha256")))
    if len(ids) != len(set(ids)) or ids != [r["sample_id"] for r in rows]:
        raise NotReady(f"FEATURE_ID_ORDER_OR_FULL_COVERAGE_MISMATCH: {path}")
    if views != [r["view"]["causal_view_sha256"] for r in rows]:
        raise NotReady(f"CAUSAL_VIEW_BINDING_MISMATCH: {path}")
    for name, array in values.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise NotReady(f"NONFINITE_NATIVE_FEATURE: {name} in {path}")
    if config["method_key"] == "ars" and split == "train":
        variant_qids = list(map(str, values["variant_qid"]))
        if not set(variant_qids) <= {r["qid"] for r in rows}:
            raise NotReady("ARS_VARIANTS_CROSS_TRAIN_BOUNDARY")
        if len(variant_qids) != len(values["variants_x"]):
            raise NotReady("ARS_VARIANT_ALIGNMENT_MISMATCH")
    labels = [int(r["label"]) for r in rows]
    if not rows:
        raise NotReady(f"NO_ACCEPTED_{split.upper()}_ROWS")
    return values, labels, rows, file_sha(path)
