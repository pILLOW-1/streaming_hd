"""Compare the original A2 VAL scores with the CPU reconstruction of A1."""

import csv
import json
from pathlib import Path
from statistics import median

from run import VOLUME, save_json


def load(path):
    return json.loads(path.read_text())


def main():
    results = VOLUME / "results"
    original = results / "qwen_c18_error_memory_20261114"
    followup = results / "qwen_c18_error_memory_followup_20261115"
    fields = ("domain", "layer", "a1_epoch", "a2_epoch", "a1_prefix_auroc",
              "a2_prefix_auroc", "a2_minus_a1")
    rows = []
    summary = {}
    for domain in ("Logic", "Factuality"):
        a1_dir = followup / domain / "A1 reproduced"
        a2_dir = original / domain / "A2"
        deltas = []
        for layer in range(1, 65):
            name = f"layer_{layer:02d}.json"
            a1, a2 = load(a1_dir / name), load(a2_dir / name)
            assert (a1["domain"], a1["arm"], a1["layer"]) == (domain, "A1 reproduced", layer)
            assert (a2["domain"], a2["arm"], a2["layer"]) == (domain, "A2", layer)
            left, right = (x["metrics"]["prefix_AUROC"] for x in (a1, a2))
            delta = right - left
            deltas.append(delta)
            rows.append((domain, layer, a1["epoch"], a2["epoch"], left, right, delta))
        summary[domain] = {
            "a2_higher": sum(delta > 0 for delta in deltas),
            "equal": sum(delta == 0 for delta in deltas),
            "a1_higher": sum(delta < 0 for delta in deltas),
            "median_delta": median(deltas),
        }
    output = followup / "paired_val_prefix.csv"
    pending = output.with_suffix(".pending")
    with pending.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    pending.replace(output)
    save_json(followup / "paired_val_prefix.json", summary)


if __name__ == "__main__":
    root = VOLUME / "results/qwen_c18_error_memory_followup_20261115"
    required = [root / domain / "TEST_COMPLETE.json" for domain in ("Logic", "Factuality")]
    if not all(path.is_file() for path in required):
        print("Waiting for both frozen TEST receipts", flush=True)
        raise SystemExit(75)
    lock = root / ".aggregate_in_progress"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(75)
    try:
        main()
    finally:
        lock.rmdir()
