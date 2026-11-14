"""Render aggregate, verified result tables for the canonical Git checkout."""
import json
from pathlib import Path
import sys


def render(report):
    domains = ("Logic", "Factuality")
    configs = ("C17", "C18", "C18 independent")
    text = ["## Results", "", "AUROC values are computed from saved predictions on identical position IDs.",
            "Recovery uses Y=0 and an earlier observed C=1.", "",
            "| Domain | Step layer / epoch | Conditional Prefix layer / epoch |",
            "|---|---:|---:|"]
    for domain in domains:
        row = report[domain]["selection"]
        text.append(f"| {domain} | {row['step']['layer']} / {row['step']['epoch']} | "
                    f"{row['kappa']['layer']} / {row['kappa']['epoch']} |")
    for split in ("val", "test"):
        for label, metric in (("Step", "step"), ("Prefix", "prefix"),
                              ("Conditional recovery macro", "kappa"), ("Prefix recovery macro", "prefix_recovery")):
            text += ["", f"### {split.upper()} {label}", "",
                     "| Configuration | Logic | Factuality |", "|---|---:|---:|"]
            for arm in configs:
                values = [report[d][split][arm][metric] for d in domains]
                if isinstance(values[0], dict):
                    values = [value["macro"] for value in values]
                text.append(f"| {arm} | {values[0]:.4f} | {values[1]:.4f} |")
    return "\n".join(text) + "\n"


def main():
    verified, output = map(Path, sys.argv[1:])
    report = json.loads(verified.read_text())
    assert set(report) == {"Logic", "Factuality"}
    readme = output / "README.md"
    intro = readme.read_text(encoding="utf-8-sig").split("\n## Results", 1)[0]
    readme.write_text(intro.replace("Status: pending.", "Status: verified on VAL and TEST.").rstrip()
                      + "\n\n" + render(report))
    result = output / "result"
    result.mkdir(exist_ok=True)
    (result / "index.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
