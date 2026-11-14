"""Render aggregate verified results for publication."""
import json
import sys
from pathlib import Path

out, private = map(Path, sys.argv[1:])
domains = ("Logic", "Factuality")
reports = {d: json.loads((private / d / "verified.json").read_text()) for d in domains}
lines = ["## Results", "", "Frozen selections use the original C18 joint VAL rule.", "",
         "| Configuration | Logic layer / epoch | Factuality layer / epoch |", "|---|---:|---:|"]
base = {"A1": ((42, 12), (50, 5)), "C17": ((35, 6), (50, 5))}
for arm in ("A1", "A2", "A3", "C17"):
    cells = base.get(arm)
    if cells is None:
        cells = [(reports[d]["selection"][arm]["layer"], reports[d]["selection"][arm]["epoch"]) for d in domains]
    lines.append("| " + arm + " | " + " | ".join(f"{layer} / {epoch}" for layer, epoch in cells) + " |")
for split in ("val", "test"):
    for name, key in (("Step", "step"), ("Prefix", "prefix"), ("Conditional recovery", "kappa"),
                      ("Prefix recovery", "prefix_recovery")):
        lines.extend(["", f"### {split.upper()} {name} AUROC", "",
                      "| Configuration | Logic | Factuality |", "|---|---:|---:|"])
        for arm in ("A1", "A2", "A3", "C17"):
            values = [reports[d][split][arm][key] for d in domains]
            if isinstance(values[0], dict):
                values = [x["macro"] for x in values]
            lines.append("| " + arm + " | " + " | ".join(f"{x:.4f}" for x in values) + " |")
readme = out / "README.md"
readme.write_text(readme.read_text(encoding="utf-8-sig").replace("Status: pending.", "Status: independently verified on VAL and TEST.").rstrip()
                  + "\n\n" + "\n".join(lines) + "\n")
result = out / "result"
result.mkdir()
(result / "index.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
