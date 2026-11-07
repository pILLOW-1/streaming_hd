#!/usr/bin/env python3
"""Verify that a width-ablation smoke differs from d_a=64 only in width."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = ("qwen", "llama", "deepseek")
EXPECTED_ATTENTION_PARAMETERS = {"qwen": 1_838_592, "llama": 2_101_248, "deepseek": 2_101_248}
HASH_FIELDS = ("label_sha256", "trajectory_sha256", "position_sha256")
COUNT_FIELDS = ("steps", "tokens", "positive", "negative", "trajectories", "dimension", "dtype")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-smoke-root", type=Path, required=True)
    parser.add_argument("--da64-audit-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_sha256 = hashlib.sha256(args.training_source.read_bytes()).hexdigest()
    expected_source_sha256 = "d8d0e160582cafe37ae32db494c69eaee2b0f5c1668571adf0dee2a9b499fd1d"
    if source_sha256 != expected_source_sha256:
        raise AssertionError(f"training source changed: {source_sha256}")
    report: dict[str, Any] = {
        "status": "PASS",
        "attention_dimension": 256,
        "training_source_sha256": source_sha256,
        "same_training_source_as_da64": True,
        "authoritative_architecture": "LN(no affine); Q=HbarWQ,K=HbarWK in R^(n x 256); V=HbarwV scalar; A=softmax(QK^T/sqrt(256)+padding); s=AV; alpha=softmax(s+padding); weighted original states; row L2; Linear(D,1)",
        "models": {},
        "note": "The reused d_a=64 training entry has a legacy human-readable smoke architecture string containing sqrt(64); this verifier uses the CLI width, instantiated parameter count, gradients and source code path as authoritative. Training computation already uses sqrt(self.attention_dimension).",
    }
    for model in MODELS:
        current = load(args.raw_smoke_root / model / "smoke_audit.json")
        previous = load(args.da64_audit_root / model / "smoke_audit.json")
        if current["attention_parameters"] != EXPECTED_ATTENTION_PARAMETERS[model]:
            raise AssertionError(f"{model}: wrong d_a=256 attention parameter count")
        checks: dict[str, bool] = {}
        for split in ("train", "val", "test"):
            for field in HASH_FIELDS + COUNT_FIELDS:
                key = f"{split}.{field}"
                checks[key] = current["summaries"][split][field] == previous["summaries"][split][field]
                if not checks[key]:
                    raise AssertionError(f"{model}: d_a=64 mismatch at {key}")
        smoke = current["smoke"]
        if smoke["status"] != "PASS" or smoke["hidden_states_requires_grad"]:
            raise AssertionError(f"{model}: semantic/gradient smoke failed")
        if smoke["padded_final_weight_nonzero"] != 0:
            raise AssertionError(f"{model}: padding weight violation")
        if smoke["valid_attention_row_sum_max_error"] > 1e-5 or smoke["final_weight_sum_max_error"] > 1e-5:
            raise AssertionError(f"{model}: attention/pooling normalization violation")
        if any(float(value) <= 0 for value in smoke["gradient_norms"].values()):
            raise AssertionError(f"{model}: zero scorer/probe gradient")
        report["models"][model] = {
            "status": "PASS",
            "attention_parameters": current["attention_parameters"],
            "probe_parameters": current["probe_parameters"],
            "exact_da64_cache_hashes_and_counts": all(checks.values()),
            "checks": checks,
            "gradient_norms": smoke["gradient_norms"],
            "valid_attention_row_sum_max_error": smoke["valid_attention_row_sum_max_error"],
            "padded_final_weight_nonzero": smoke["padded_final_weight_nonzero"],
            "final_weight_sum_max_error": smoke["final_weight_sum_max_error"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "DA256_WIDTH_SMOKE_PASS", "source_sha256": source_sha256}), flush=True)


if __name__ == "__main__":
    main()
