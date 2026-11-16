"""Score TRAIN with the frozen legacy fixed-aggregation Step detector."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from train import checked, metrics, read_json, saved_step_scores, test_split
from run import Split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=("Logic", "Factuality"))
    parser.add_argument("source", type=Path)
    parser.add_argument("pins", type=Path)
    parser.add_argument("history_index", type=Path)
    parser.add_argument("input_manifest_sha")
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = read_json(checked(args.source / "input/manifest.json", args.input_manifest_sha))
    domain = args.domain
    tag = domain + "/streaming/"
    row = manifest["cells"][domain]["streaming"]
    assert row["step_source"] == "streaming_fixed"
    files = {n: checked(args.source / "input" / (tag + n), manifest["files"][tag + n])
             for n in ("frozen_step.pt", "step_val.npz", "step_test.npz")}
    pins = read_json(args.pins)[domain]
    train, val = (Split(domain, split, pins[split]) for split in ("train", "val"))
    test = test_split(domain, args.history_index)
    for split, data in (("val", val), ("test", test)):
        scores = saved_step_scores(files[f"step_{split}.npz"], data, "streaming", device)
        auroc = metrics(data.y, scores.cpu().numpy())["AUROC"]
        assert abs(auroc - row[f"step_{split}_auroc"]) < 1e-10, (split, auroc)
    code = Path(__file__).with_name("fixed_step_code.py")
    spec = importlib.util.spec_from_file_location("fixed_step_code", code)
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    out = args.source / tag / "train_step.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with np.load(out, allow_pickle=False) as saved:
            assert saved["frozen_sha"].item() == manifest["files"][tag+"frozen_step.pt"]
            assert saved["positions_sha"].item() == train.pins["positions"]
        return
    checkpoint = torch.load(files["frozen_step.pt"], map_location="cpu", weights_only=True)
    assert checkpoint["selection"]["layer"] == row["layer"]
    assert checkpoint["selection"]["epoch"] == row["epoch"]
    model = fixed.Detector(fixed.FIXED_ARM).to(device)
    model.load_state_dict(checkpoint["model"])
    q = fixed.predict(model, fixed.Layer(train, row["layer"]), device)
    assert q.shape == train.y.shape and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
    pending = out.with_suffix(".pending.npz")
    np.savez_compressed(pending, q=q, frozen_sha=manifest["files"][tag+"frozen_step.pt"],
                        positions_sha=train.pins["positions"])
    pending.replace(out)
    print("FROZEN_FIXED_STEP_READY", domain, row["layer"], len(q), flush=True)


if __name__ == "__main__":
    main()
