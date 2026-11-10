#!/usr/bin/env python3
"""Build manuscript comparison tables from independently verified TEST results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

MODELS = [
    ("deepseek_r1_distill_llama_8b", "DS-L8"),
    ("gemma_4_31b_it", "Gemma-4-31B-it"),
    ("qwen3_8_27b", "Qwen3.8-27B"),
]
DOMAINS = ["Math", "Science", "Logic", "Multi-hop", "Factuality"]
METHODS = {
    "step": [("saplma", "SAPLMA"), ("hami", "HaMI"), ("global_mean", "Global Mean"), ("streaming", "Streaming"), ("ours", "Ours")],
    "prefix": [("saplma", "SAPLMA"), ("icr_probe", "ICR Probe"), ("llm_check", "LLM-Check"), ("global_mean", "Global Mean"), ("streaming", "Streaming"), ("ours", "Ours")],
    "answer": [("saplma", "SAPLMA"), ("ars", "ARS"), ("rhd", "RHD"), ("race", "RACE"), ("rede", "ReDe"), ("global_mean", "Global Mean"), ("streaming", "Streaming"), ("ours", "Ours")],
}


def build(root: Path) -> None:
    """Read accepted public result artifacts; never substitute VAL or partial fits."""
    values = {}
    sources = []

    def add(method: str, record: dict, metrics: dict, source: Path) -> None:
        key = (record["level"], method, record["model"], record["domain"])
        assert key not in values, f"Duplicate result: {key}"
        if method == "streaming" and record["level"] == "answer":
            # The archived Answer schema records one prediction per question,
            # with coverage expressed through the complete confusion counts.
            assert sum(metrics[k] for k in ["TP", "FP", "TN", "FN"]) == metrics["questions"]
        else:
            assert metrics["missing"] == 0, f"Incomplete TEST: {source}"
        value = float(metrics["AUROC"])
        assert 0 <= value <= 1, f"Invalid AUROC: {source}"
        values[key] = value
        sources.append({"level": key[0], "method": method, "model": key[2], "domain": key[3], "AUROC": value, "source": source.relative_to(root).as_posix(), "SHA256": hashlib.sha256(source.read_bytes()).hexdigest()})

    for source in sorted((root / "experiments").glob("*/622_p70_20261108/result/*/*/*.json")):
        record = json.loads(source.read_text())
        if record.get("validation_status") != "independently checked from saved TEST predictions":
            continue
        add(record["method"], record, record["TEST"], source)
    source = root / "experiments/streaming_622_p70/result/metrics.json"
    records = json.loads(source.read_text())
    assert len(records) == 45, "Expected the retained 45-cell Streaming reproduction"
    for record in records:
        add("streaming", record, record, source)

    output = root / "reports/main_result"
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    rows = []
    for level, methods in METHODS.items():
        best = {}
        for model, _ in MODELS:
            for domain in DOMAINS:
                available = [values[(level, method, model, domain)]
                             for method, _ in methods
                             if (level, method, model, domain) in values]
                if available:
                    best[(model, domain)] = max(available)
        lines = [f"<h2>{level.title()}-level (AUROC)</h2>", "<table>", '<thead><tr><th rowspan="2">Method</th>']
        lines += [f'<th colspan="5">{html.escape(name)}</th>' for _, name in MODELS]
        lines += ["</tr><tr>"]
        lines += [f"<th>{'Fact' if d == 'Factuality' else d}</th>" for _ in MODELS for d in DOMAINS]
        lines += ["</tr></thead>", "<tbody>"]
        for method, label in methods:
            lines.append(f"<tr><td>{label}</td>")
            for model, _ in MODELS:
                for domain in DOMAINS:
                    value = values.get((level, method, model, domain))
                    if value is None:
                        lines.append("<td></td>")
                    else:
                        text = f"{100 * value:.2f}"
                        if value == best[(model, domain)]:
                            text = f"<strong>{text}</strong>"
                        lines.append(f"<td>{text}</td>")
                    rows.append({"level": level, "method": label, "model": model, "domain": domain, "AUROC_percent": f"{100 * value:.2f}" if value is not None else ""})
            lines.append("</tr>")
        lines += ["</tbody></table>"]
        tables.append("\n".join(lines))
    intro = "# Main comparison\n\n6:2:2, TEST target p ≥ 70% (frozen per-combination relaxations retained).\n\nAUROC (%), higher is better. Bold marks the highest available AUROC in each column within each level, compared before rounding; exact ties are all bold. Only completed, independently verified TEST results are shown. Blank cells indicate unavailable results; Ours is not finalized. Layers and weights are selected on VAL.\n\n"
    (output / "comparison.md").write_text(intro + "\n\n".join(tables) + "\n")
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    expected = {(level, method, model, domain) for level, methods in METHODS.items() for method, _ in methods for model, _ in MODELS for domain in DOMAINS}
    assert set(values) <= expected, "Unexpected result identity"
    assert sum(row["AUROC_percent"] != "" for row in rows) == len(values)
    print(f"Comparison: {len(values)} completed cells; {len(rows) - len(values)} blank cells (including Ours).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    build(parser.parse_args().repo_root.resolve())
