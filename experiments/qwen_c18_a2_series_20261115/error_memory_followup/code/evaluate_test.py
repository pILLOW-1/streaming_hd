"""Evaluate the frozen independent pair on TEST using the existing feature bank."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import make_model
from run import VOLUME, Recovery, Split, digest, evaluate, save_json, score


def baseline_root(domain):
    results = VOLUME / "results"
    if domain == "Logic":
        return results / "logic_ordered_dual" / domain / "ordered_dual" / "linear"
    return results / "factuality_c01_c20_20261112" / "dual" / domain / "ordered_dual" / "linear"


def baseline(domain, subdir, data, recovery):
    path = baseline_root(domain) / subdir
    report = json.loads((path / "TEST_RESULT.json").read_text())
    predictions = path / "test_predictions.npz"
    pin = digest(predictions.read_bytes())
    assert report["predictions_SHA256"] == pin and report["domain"] == domain
    with np.load(predictions, allow_pickle=False) as saved:
        lookup = {key: i for i, key in enumerate(saved["position_id"])}
        assert len(lookup) == len(saved["position_id"]) and set(lookup) == set(data.keys)
        order = [lookup[k] for k in data.keys]
        assert np.array_equal(data.y, saved["y"][order])
        assert np.array_equal(data.c, saved["c"][order])
        p = saved["probability"][order].astype(np.float64)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    kappa = p[:, 1] / p[:, 1:].sum(axis=1)
    return dict(metrics=evaluate(data, recovery, p[:, 0], kappa, p[:, :2].sum(axis=1)),
                layer=report["selected_layer"], epoch=report["selected_epoch"], prediction_sha256=pin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    parser.add_argument("output", type=Path)
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    domain, output = args.domain, args.output
    torch.set_num_threads(2)
    run_info = json.loads((output / "selection.json").read_text())
    report = json.loads((output / "result.json").read_text())
    assert run_info["completed_layers"] == 64 and run_info["max_epoch"] == 100
    assert report["domain"] == domain
    index_bytes = args.index.read_bytes()
    index = json.loads(index_bytes)
    # The index carries only token spans and identity. Reuse verified TEST labels already on this volume.
    baseline_path = baseline_root(domain) / "ce"
    baseline_file = baseline_path / "test_predictions.npz"
    original = json.loads((baseline_path / "TEST_RESULT.json").read_text())
    assert original["predictions_SHA256"] == digest(baseline_file.read_bytes())
    with np.load(baseline_file, allow_pickle=False) as saved:
        labels = {key: (int(y), int(c)) for key, y, c in zip(
            saved["position_id"], saved["y"], saved["c"], strict=True)}
    rows, used = [], set()
    for row in index["rows"]:
        sample = row["sample_id"]
        selected = [(i, span, *labels[f"{sample}:{i}"])
                    for i, span in enumerate(row["step_token_spans"]) if f"{sample}:{i}" in labels]
        steps, spans, y, c = list(zip(*selected, strict=True)) if selected else ([], [], [], [])
        used.update(f"{sample}:{i}" for i in steps)
        rows.append(dict(sample_id=sample, token_sha=row["token_sha"], origin=row["origin"],
                         retained_original_step_ids=steps, step_token_spans=spans,
                         step_hallucination=y, prefix_hallucination=c))
    assert used == set(labels)
    data = Split(domain, "test", dict(full_sha=index["full_sha"]), rows=rows)
    recovery = Recovery(data)
    comparisons = {arm: baseline(domain, subdir, data, recovery)
                   for arm, subdir in (("C17", "marginal_bce"), ("C18", "ce"))}
    if (output / "test.json").exists():
        previous = json.loads((output / "test.json").read_text())
        assert previous["comparisons"]["C18 independent"]["prediction_sha256"] == digest(
            (output / "test.npz").read_bytes())
        return
    # Create the guard before any TEST inference.
    with (output / "TEST_STARTED.json").open("x") as guard:
        json.dump(dict(domain=domain, selection=report["selection"],
                       index_sha256=digest(index_bytes), compared_runs=comparisons), guard)
    scores = {}
    for name, method in (("step", "step_exp"), ("kappa", "prefix_exp")):
        pin = report["selection"][name]
        checkpoint = output / pin["file"]
        assert digest(checkpoint.read_bytes()) == pin["sha256"]
        model = make_model(5120, "linear")
        head = model.step if name == "step" else model.conditional_prefix
        head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        batch, = data.features(pin["layer"], (method,))
        scores[name] = score(head, batch)
    q, kappa = scores["step"], scores["kappa"]
    pred = output / "test.npz"
    np.savez_compressed(pred, q=q, kappa=kappa, position_id=data.keys, group=data.group,
                        prior=data.prior, y=data.y, c=data.c)
    comparisons["C18 independent"] = dict(metrics=evaluate(data, recovery, q, kappa),
                                          prediction_sha256=digest(pred.read_bytes()))
    save_json(output / "test.json", dict(domain=domain, coverage=recovery.coverage,
                                          comparisons=comparisons, inputs=data.pins))
    print("TEST_COMPLETE", domain, flush=True)


if __name__ == "__main__":
    main()
