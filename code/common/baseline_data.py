"""Explicit accepted-version, split, independent-label and causal-view contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


class NotReady(ValueError):
    """An exact prerequisite is missing or disagrees with the accepted contract."""


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def checked_asset(item: dict, base: Path) -> Path:
    path = Path(item["path"])
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise NotReady(f"MISSING_ASSET: {path}")
    if file_sha(path) != item["sha256"]:
        raise NotReady(f"ASSET_HASH_MISMATCH: {path}")
    from common.baseline_io import stage_immutable
    return stage_immutable(path, item['sha256'])


def accepted_manifest(path: Path) -> dict:
    if not path.is_file():
        raise NotReady(f"MISSING_ACCEPTED_DATASET_MANIFEST: {path}")
    value = json.loads(path.read_text())
    required = {"dataset_id", "version", "status", "label_version", "schema_version",
                "trajectories", "split_mapping", "acceptance", "source_manifests",
                "coverage", "causal_boundary_contract"}
    missing = required - value.keys()
    if missing:
        raise NotReady(f"MANIFEST_FIELDS_MISSING: {sorted(missing)} in {path}")
    if value["status"] != "ACCEPTED" or value["schema_version"] != "baseline_trajectory_v1":
        raise NotReady(f"DATASET_NOT_ACCEPTED_OR_SCHEMA_UNBOUND: {path}")
    if value["label_version"] not in {"original", "cleaned", "revised"}:
        raise NotReady("label_version must explicitly be original, cleaned, or revised")
    receipt_path = checked_asset(value["acceptance"], path.parent)
    receipt = json.loads(receipt_path.read_text())
    for field in ("dataset_id", "version", "label_version", "schema_version"):
        if receipt.get(field) != value[field]:
            raise NotReady(f"ACCEPTANCE_IDENTITY_MISMATCH: {field}")
    if receipt.get("status") != "PASS_ACCEPTED_DATASET":
        raise NotReady(f"ACCEPTANCE_GATE_NOT_PASS: {receipt_path}")
    if value["causal_boundary_contract"] != "question_plus_visible_content_no_future_no_gold":
        raise NotReady("CAUSAL_BOUNDARIES_NOT_ACCEPTED")
    for asset in [value["trajectories"], value["split_mapping"], *value["source_manifests"]]:
        checked_asset(asset, path.parent)
    for key in ("eligible", "excluded", "unknown"):
        if key not in value["coverage"]:
            raise NotReady(f"COVERAGE_DENOMINATOR_MISSING: {key}")
    value["_path"] = str(path.resolve())
    value["_sha256"] = file_sha(path)
    return value


def split_mapping(manifest: dict) -> dict[tuple[str, str], str]:
    base = Path(manifest["_path"]).parent
    path = checked_asset(manifest["split_mapping"], base)
    result: dict[tuple[str, str], str] = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if not item.get("generator"):
            raise NotReady("SPLIT_GENERATOR_ID_REQUIRED; do not overwrite model-specific original splits")
        key, split = (item["generator"], str(item["qid"])), item["split"]
        if split not in {"train", "val", "test"} or key in result:
            raise NotReady(f"DUPLICATE_OR_INVALID_ORIGINAL_QUESTION_SPLIT: {key}")
        result[key] = split
    if not all(name in result.values() for name in ("train", "val", "test")):
        raise NotReady("ALL_THREE_ORIGINAL_SPLITS_REQUIRED; never repartition here")
    return result


def validate_trajectory(item: dict) -> None:
    fields = {"tid", "qid", "generator", "domain", "question", "options_or_context",
              "generated_answer", "original_steps", "retained_original_step_ids",
              "step_labels", "prefix_labels", "answer_label", "source_sha256",
              "label_version", "annotation_status"}
    if fields - item.keys():
        raise NotReady(f"TRAJECTORY_SCHEMA_FIELDS_MISSING: {sorted(fields-item.keys())}")
    ids = item["retained_original_step_ids"]
    if ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < len(item["original_steps"]) for i in ids):
        raise NotReady(f"ORIGINAL_STEP_MAPPING_INVALID: {item['tid']}")
    if len(ids) != len(item["step_labels"]) or len(ids) != len(item["prefix_labels"]):
        raise NotReady(f"INDEPENDENT_LABEL_ALIGNMENT_INVALID: {item['tid']}")
    labels = item["step_labels"] + item["prefix_labels"] + [item["answer_label"]]
    if any(v is not None and (type(v) is not int or v not in (0, 1)) for v in labels):
        raise NotReady("BINARY_HALLUCINATION_LABEL_1_REQUIRED; unknown must be explicit null")


def causal_view(item: dict, level: str, position: int | None) -> dict:
    """Whitelist the visible text; neither labels nor gold/future fields enter a view."""
    validate_trajectory(item)
    head = item["question"]
    if item["options_or_context"]:
        head += "\n\n" + item["options_or_context"]
    prompt = head + "\n\n"
    if level in {"step", "prefix"}:
        if position is None or not 0 <= position < len(item["retained_original_step_ids"]):
            raise NotReady("STEP_POSITION_REQUIRED")
        original_id = item["retained_original_step_ids"][position]
        # Removed single-symbol steps remain in the causal context, preserving
        # the original generation, while their labels never become training rows.
        visible = item["original_steps"][:original_id + 1]
        body = "\n\n".join(visible)
        current_start = len(prompt) + len("\n\n".join(visible[:-1])) + (2 if len(visible) > 1 else 0)
        target_span = [current_start, len(prompt + body)]
    elif level == "answer":
        if position is not None:
            raise NotReady("ANSWER_HAS_NO_STEP_POSITION")
        reasoning = "\n\n".join(item["original_steps"])
        body = reasoning + ("\n\n" if reasoning else "") + item["generated_answer"]
        target_span = [len(prompt) + len(reasoning) + (2 if reasoning else 0), len(prompt + body)]
        original_id = None
    else:
        raise NotReady(f"UNSUPPORTED_LEVEL: {level}")
    view = {"text": prompt + body, "question_end": len(head), "reasoning_start": len(prompt),
            "target_char_span": target_span, "visible_end": len(prompt + body),
            "original_step_id": original_id, "level": level}
    view["causal_view_sha256"] = canonical_sha(view)
    return view


def iter_samples(item: dict, level: str, split: str) -> Iterator[dict]:
    validate_trajectory(item)
    positions = [None] if level == "answer" else range(len(item["retained_original_step_ids"]))
    for position in positions:
        label = item["answer_label"] if level == "answer" else item[level + "_labels"][position]
        original_id = None if position is None else item["retained_original_step_ids"][position]
        status_key = "answer" if position is None else f"{level}:{original_id}"
        status = item["annotation_status"].get(status_key)
        uid = canonical_sha([item["tid"], level, original_id, item["source_sha256"], item["label_version"]])
        yield {"sample_id": uid, "tid": item["tid"], "qid": str(item["qid"]),
               "generator": item["generator"], "domain": item["domain"], "split": split,
               "level": level, "position": position, "original_step_id": original_id,
               "label": label, "eligible": label is not None and status == "ACCEPTED",
               "annotation_status": status, "view": causal_view(item, level, position)}


def dataset_samples(manifest: dict, level: str, generator: str, allowed_splits=None) -> Iterator[dict]:
    splits = split_mapping(manifest)
    path = checked_asset(manifest["trajectories"], Path(manifest["_path"]).parent)
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        item = json.loads(line)
        if item["generator"] != generator:
            continue
        key = (item["generator"], str(item["qid"]))
        if item["tid"] in seen or key not in splits:
            raise NotReady("DUPLICATE_TRAJECTORY_OR_UNFROZEN_QUESTION_SPLIT")
        if item["label_version"] != manifest["label_version"]:
            raise NotReady("SILENT_LABEL_VERSION_SWITCH_FORBIDDEN")
        seen.add(item["tid"])
        if allowed_splits is not None and splits[key] not in allowed_splits:
            continue
        yield from iter_samples(item, level, splits[key])


def normalize_cetus_source(source: dict, label_version: str, approved_level_status: dict, revised=None) -> dict:
    """Thin schema adapter for the actual Cetus review trajectory structure.

    Acceptance, label version and per-level eligibility must come from the final
    dataset producer, not inferred from a live review status or an OR of Steps.
    """
    if label_version == "revised":
        if revised is None or revised["tid"] != source["tid"] or revised["source_sha256"] != source["source_sha256"]:
            raise NotReady("EXACT_REVISED_TO_SOURCE_JOIN_REQUIRED")
        chosen = revised
    elif label_version in {"original", "cleaned"}:
        if revised is not None:
            raise NotReady("EXPLICIT_LABEL_VERSION_MISMATCH")
        chosen = source
    else:
        raise NotReady("EXPLICIT_LABEL_VERSION_REQUIRED")
    keys = ("tid", "qid", "generator", "domain", "question", "options_or_context",
            "generated_answer", "original_steps", "retained_original_step_ids", "source_sha256")
    result = {key: source[key] for key in keys}
    result.update({key: chosen[key] for key in ("step_labels", "prefix_labels", "answer_label")})
    result.update(label_version=label_version, annotation_status=dict(approved_level_status))
    validate_trajectory(result)
    return result
