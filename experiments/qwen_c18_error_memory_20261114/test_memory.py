"""Evaluate each joint VAL-selected memory detector once on frozen TEST."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate_test import baseline, baseline_root
from history import HistorySplit
from memory_model import MemoryHeads
from run import Recovery, digest, save_json
from train_memory import ARMS, evaluate_prob, predict, rank


def main(domain, root, index_path):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert json.loads((root / "TRAIN_COMPLETE.json").read_text()) == {
        "layers": 64, "arms": list(ARMS), "domain": domain}
    index = json.loads(index_path.read_text())
    reference = baseline_root(domain) / "ce"
    data = (reference / "test_predictions.npz").read_bytes()
    assert digest(data) == json.loads((reference / "TEST_RESULT.json").read_text())["predictions_SHA256"]
    with np.load(reference / "test_predictions.npz", allow_pickle=False) as saved:
        labels = {k: (int(y), int(c)) for k, y, c in zip(
            saved["position_id"], saved["y"], saved["c"], strict=True)}
    rows, used = [], set()
    for row in index["rows"]:
        sample = row["sample_id"]
        selected = [(i, span, *labels[f"{sample}:{i}"])
                    for i, span in enumerate(row["step_token_spans"]) if f"{sample}:{i}" in labels]
        steps, spans, y, c = zip(*selected) if selected else ([], [], [], [])
        used.update(f"{sample}:{i}" for i in steps)
        rows.append(dict(sample_id=sample, token_sha=row["token_sha"], origin=row["origin"],
                         retained_original_step_ids=steps, step_token_spans=spans,
                         step_hallucination=y, prefix_hallucination=c))
    assert used == set(labels)
    test = HistorySplit(domain, "test", {"full_sha": index["full_sha"]}, index_path, labels=rows)
    recovery = Recovery(test)
    references = {arm: baseline(domain, loss, test, recovery)
                  for arm, loss in (("A1", "ce"), ("C17", "marginal_bce"))}

    selections = {}
    for arm in ARMS:
        out = root / arm
        selected = json.loads((out / "selection.json").read_text())
        candidates = [json.loads((out / f"layer_{layer:02d}.json").read_text()) for layer in range(1, 65)]
        assert [c["layer"] for c in candidates] == list(range(1, 65))
        best = max(candidates, key=lambda row: rank(row["metrics"]))
        assert all(selected[k] == best[k] for k in ("arm", "domain", "layer", "epoch", "metrics"))
        selections[arm] = selected

    device = torch.device("cuda")
    for layer in sorted({s["layer"] for s in selections.values()}):
        pending = [a for a in ARMS if selections[a]["layer"] == layer and not (root / a / "test.json").exists()]
        if not pending:
            continue
        features = test.layer(layer, device)
        for arm in pending:
            out, pin = root / arm, selections[arm]
            ckpt = out / pin["checkpoint"]
            assert digest(ckpt.read_bytes()) == pin["checkpoint_sha256"]
            model = MemoryHeads(5120, arm).to(device)
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
            with (out / "TEST_STARTED.json").open("x") as guard:
                json.dump(dict(selection=pin, inputs=test.pins), guard)
            p = predict(model, features)
            path = out / "test.npz"
            np.savez_compressed(path, probability=p, position_id=test.keys, y=test.y, c=test.c,
                                group=test.group, prior=test.prior)
            save_json(out / "test.json", dict(domain=domain, arm=arm, metrics=evaluate_prob(test, p),
                                              coverage=recovery.coverage, inputs=test.pins,
                                              prediction_sha256=digest(path.read_bytes()), references=references))
            print("TEST_COMPLETE", domain, arm, flush=True)
    save_json(root / "TEST_COMPLETE.json", dict(domain=domain, arms=ARMS))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
