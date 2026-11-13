"""Render public metrics from the eight verified VAL replay receipts."""
import csv
import hashlib
import json
import sys
from pathlib import Path

DOMAINS = ("Logic", "Factuality")
ARMS = ("C17", "C18", "C19", "C20")
SCORES = ("prefix", "kappa")
METRICS = ("pooled", "within_pair", "within_macro")


def main():
    receipts, destination = map(Path, sys.argv[1:])
    result = destination / "result"
    result.mkdir(parents=True, exist_ok=True)
    entries = []
    for domain in DOMAINS:
        for arm in ARMS:
            name = f"{domain}.{arm}"
            row = json.loads((receipts / f"{name}.json").read_text())
            assert (row["domain"], row["arm"]) == (domain, arm)
            pred = receipts / f"{name}.val.npz"
            assert hashlib.sha256(pred.read_bytes()).hexdigest() == row["prediction_sha256"]
            assert max(map(abs, row["replay_val_metric_deltas"].values())) <= 1e-4
            entries.append(row)
    (result / "index.json").write_text(json.dumps(
        dict(status="VAL_DIAGNOSTIC_VERIFIED", configurations=entries),
        indent=2, allow_nan=False) + "\n")
    with (result / "auroc.csv").open("w", newline="") as stream:
        fields = list(entries[0]["scores"]["prefix"]["recovered"])
        writer = csv.DictWriter(stream, fieldnames=[
            "domain", "configuration", "layer", "score", "comparison", *fields], lineterminator="\n")
        writer.writeheader()
        for row in entries:
            for score, comparisons in row["scores"].items():
                for comparison, metrics in comparisons.items():
                    writer.writerow(dict(domain=row["domain"], configuration=row["arm"],
                                         layer=row["layer"], score=score,
                                         comparison=comparison, **metrics))
    readme = Path(__file__).with_name("README.md").read_text().replace(
        "Execution pending.", "All eight configurations passed the frozen VAL metric check.")
    readme += (
        "\nEach cell shows pooled AUROC / within-trajectory pair AUROC / "
        "equal-weight trajectory macro AUROC. Each score uses the same eligible positions. "
        "The CSV records mixed-label trajectory counts, positive and negative counts, "
        "and the number of within-trajectory pairs.\n"
    )
    lookup = {(r["domain"], r["arm"]): r for r in entries}
    for score in SCORES:
        for comparison in ("recovered", "fresh"):
            title = "Prefix" if score == "prefix" else "Conditional Prefix (κ)"
            readme += f"\n## {title} against {comparison} clean\n\n"
            readme += "| Configuration | Logic | Factuality |\n|---|---:|---:|\n"
            for arm in ARMS:
                cells = []
                for domain in DOMAINS:
                    values = lookup[domain, arm]["scores"][score][comparison]
                    cells.append(" / ".join(f"{values[m]:.4f}" if values[m] is not None else ""
                                            for m in METRICS))
                readme += f"| {arm} | {' | '.join(cells)} |\n"
    (destination / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
