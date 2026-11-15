"""Evaluate the global VAL-selected detector once on the frozen TEST split."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate_test import baseline_root
from history import HistorySplit
from run import digest, save_json
from train import Detector, Layer, predict, rank, val_stats


def main(domain, arm, root, index_path, device_name):
    halves = [root / side for side in ("lower", "upper")]
    if not all((half / "SHARD_COMPLETE.json").is_file() for half in halves):
        return
    if (root / "TEST_COMPLETE.json").is_file():
        return
    lock = root / ".evaluate"
    try:
        lock.mkdir()
    except FileExistsError:
        return
    try:
        if (root / "TEST_COMPLETE.json").is_file():
            return
        candidates = []
        for work, first, last in zip(halves, (1, 33), (32, 64), strict=True):
            marker = json.loads((work / "SHARD_COMPLETE.json").read_text())
            assert marker == dict(domain=domain, arm=arm, first=first, last=last)
            for layer in range(first, last + 1):
                record = json.loads((work / f"layer_{layer:02d}.json").read_text())
                assert (record["domain"], record["arm"], record["layer"]) == (domain, arm, layer)
                candidates.append((record, work))
        selected, work = max(candidates, key=lambda item: rank(item[0]["metrics"]))
        pt = work / selected["checkpoint"]
        assert digest(pt.read_bytes()) == selected["checkpoint_sha256"]
        device = torch.device(device_name)
        model = Detector(arm).to(device)
        model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True)["model"])

        baseline = baseline_root(domain) / "ce"
        reference = (baseline / "test_predictions.npz").read_bytes()
        assert digest(reference) == json.loads((baseline / "TEST_RESULT.json").read_text())["predictions_SHA256"]
        with np.load(baseline / "test_predictions.npz", allow_pickle=False) as bank:
            labels = {k: (int(y), int(c)) for k, y, c in zip(bank["position_id"], bank["y"], bank["c"], strict=True)}
        index = json.loads(index_path.read_text())
        rows, seen = [], set()
        for row in index["rows"]:
            sample = row["sample_id"]
            selected_steps = [(i, span, *labels[f"{sample}:{i}"])
                              for i, span in enumerate(row["step_token_spans"]) if f"{sample}:{i}" in labels]
            steps, spans, y, c = zip(*selected_steps) if selected_steps else ([], [], [], [])
            seen.update(f"{sample}:{i}" for i in steps)
            rows.append(dict(sample_id=sample, token_sha=row["token_sha"], origin=row["origin"],
                             retained_original_step_ids=steps, step_token_spans=spans,
                             step_hallucination=y, prefix_hallucination=c))
        assert seen == set(labels)
        data = HistorySplit(domain, "test", {"full_sha": index["full_sha"]}, index_path, labels=rows)
        features = Layer(data, selected["layer"])
        with (root / "TEST_STARTED.json").open("x") as guard:
            json.dump(dict(selection=selected, inputs=data.pins), guard)
        probability = predict(model, features, device)
        output = root / "test.npz"
        np.savez_compressed(output, probability=probability, position_id=data.keys, y=data.y)
        save_json(root / "TEST_COMPLETE.json",
                  dict(domain=domain, arm=arm, selection=selected, inputs=data.pins,
                       sha256=digest(output.read_bytes()), metrics=val_stats(data, probability)))
    finally:
        lock.rmdir()


if __name__ == "__main__":
    domain, arm, root, index, device = sys.argv[1:]
    main(domain, arm, Path(root), Path(index), device)
