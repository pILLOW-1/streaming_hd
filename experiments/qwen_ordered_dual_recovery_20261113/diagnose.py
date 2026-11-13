"""Replay frozen ordered heads on VAL and evaluate recovery discrimination."""
import ast
import hashlib
import io
import json
import sys
from pathlib import Path

import boto3
import numpy as np
import torch

PROJECT = Path("/shared/homes/u26487468/agent_uncertainty_quantification")
CODE = PROJECT.parent / "streaming_hd/code"
RUNS = PROJECT / "experiments/streaming_hd/runs"
ORIGINAL = RUNS / "step_prefix_ordered_dual_qwen_wd1e4_20261111/runpod"
OUTPUT = RUNS / "qwen_ordered_dual_recovery_20261113"
BUCKET = "fcsu28qkr3"
MODEL = "qwen3_8_27b"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
READOUTS = ("step_exp", "prefix_exp")
ARMS = {
    "C17": ("linear", "marginal_bce"),
    "C18": ("linear", "ce"),
    "C19": ("mlp", "marginal_bce"),
    "C20": ("mlp", "ce"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ranked_metrics(labels, scores):
    order = np.argsort(scores)[::-1]
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(labels)]
    true = np.cumsum(labels[order])[ends - 1]
    false = ends - true
    positives, negatives = true[-1], false[-1]
    assert positives and negatives
    gained_true = np.diff(np.r_[0, true])
    gained_false = np.diff(np.r_[0, false])
    auc = np.sum(gained_true * (negatives - false + 0.5 * gained_false)) / (positives * negatives)
    ap = np.sum(gained_true * true / ends) / positives
    return float(auc), float(ap)


def compare(labels, groups, scores):
    assert set(labels) == {0, 1}
    aucs, counts = [], []
    for group in np.unique(groups):
        selected = groups == group
        local = labels[selected]
        positive = int(local.sum())
        pairs = positive * (len(local) - positive)
        if pairs:
            counts.append(pairs)
            aucs.append(ranked_metrics(local, scores[selected])[0])
    return dict(
        pooled=ranked_metrics(labels, scores)[0],
        within_pair=float(np.average(aucs, weights=counts)) if counts else None,
        within_macro=float(np.mean(aucs)) if counts else None,
        mixed_trajectories=len(counts), eligible_trajectories=len(np.unique(groups)),
        within_pairs=sum(counts), positive=int(labels.sum()), negative=int((1-labels).sum()),
    )


def bind(domain, source, expected_hashes, load_split):
    base = PROJECT / "assets/streaming_hd_new" / MODEL / domain
    for rel, pin in expected_hashes.items():
        if rel.startswith("val/"):
            assert sha((base / rel).read_bytes()) == pin, rel
    rows = load_split(PROJECT / "assets/streaming_hd_new", MODEL, domain, "val")
    prefix = f"qwen_relation/{domain}/val/"
    label_data = source(prefix + "labels.b45b2fd.json")
    hub = json.loads(label_data)
    assert hub["domain"] == domain and hub["split"] == "val"
    assert hub["revision"] == "b45b2fd88e00869825de163bc6b7a41f31dd962b"
    by_id = {r["sample_id"]: r for r in hub["rows"]}
    full_path = PROJECT / "assets/streaming_hd_baseline_inputs/native_full" / MODEL / domain / "val.native.jsonl"
    full_data = full_path.read_bytes()
    native = {r["sample_id"]: r for r in map(json.loads, full_data.splitlines())}
    manifest_data = source(prefix + "TOKEN_WRITE_RECEIPT.json")
    receipt = json.loads(manifest_data)
    assert receipt["full_native_SHA256"] == sha(full_data)
    assert receipt["model_revision"] == receipt["tokenizer_revision"] == REVISION
    assert receipt["domain"] == domain and receipt["split"] == "val"
    assert receipt["dtype"] == "bfloat16_bits" and receipt["width"] == 5120
    offsets_data = source(prefix + "TOKEN_OFFSETS.json")
    if "offsets_SHA256" in receipt:
        assert sha(offsets_data) == receipt["offsets_SHA256"]
    offsets = json.loads(offsets_data)
    assert len(rows) == len(native) == len(by_id) == len(offsets) == receipt["rows"]
    assert {r["sample_id"] for r in rows} == set(native) == set(by_id) == {r["sample_id"] for r in offsets}
    keys, group, y, c, prior = [], [], [], [], []
    spans, slots = {}, {}
    for r in rows:
        sample = r["sample_id"]
        h = by_id[sample]
        fields = ("retained_original_step_ids", "step_hallucination", "prefix_hallucination")
        assert all(r[k] == h[k] for k in fields)
        steps, ys, cs = (r[k] for k in fields)
        n = native[sample]
        retained_spans = [n["step_token_spans"][i] for i in steps]
        assert retained_spans == h["step_token_spans"]
        spans[sample] = retained_spans
        slots[sample] = slice(len(keys), len(keys) + len(steps))
        history = False
        for step, yl, cl in zip(steps, ys, cs, strict=True):
            keys.append(f"{sample}:{step}")
            group.append(sample)
            y.append(yl)
            c.append(cl)
            prior.append(history)
            history |= bool(cl)
    assert len(keys) == len(set(keys))
    return dict(prefix=prefix, receipt=receipt, offsets=offsets, native=native,
                spans=spans, slots=slots, keys=keys, group=np.asarray(group),
                y=np.asarray(y), c=np.asarray(c), prior=np.asarray(prior),
                labels_sha256=sha(label_data), offsets_sha256=sha(offsets_data),
                token_receipt_sha256=sha(manifest_data))


def pooled(client, binding, layer, pools):
    receipt = binding["receipt"]
    width = receipt["width"]
    arrays = {m: np.empty((len(binding["keys"]), width), np.float32) for m in READOUTS}
    name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
    pin = receipt["files"][name]
    obj = client.get_object(Bucket=BUCKET, Key=binding["prefix"] + name)
    assert obj["ContentLength"] == pin["bytes"] and pin["layer"] == layer
    with obj["Body"] as stream:
        version = np.lib.format.read_magic(stream)
        read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = read_header(stream)
        assert shape == (receipt["tokens"], width) and not fortran and dtype == np.dtype("<u2")
        cursor = 0
        for offset in binding["offsets"]:
            sample = offset["sample_id"]
            n = binding["native"][sample]
            assert n["token_input_SHA256"] == offset["token_SHA256"]
            assert cursor == offset["offset"]
            tokens = offset["reasoning_tokens"] + offset["answer_tokens"]
            payload = stream.read(tokens * width * 2)
            assert len(payload) == tokens * width * 2
            bits = np.frombuffer(payload, dtype=dtype).reshape(tokens, width)
            spans = binding["spans"][sample]
            start = n["reasoning_start_token"]
            lo = start - offset["spans"][0][0]
            hi = max(b for _, b in spans) - offset["spans"][0][0]
            assert 0 <= lo < hi <= offset["reasoning_tokens"]
            visible = (bits[lo:hi].astype(np.uint32) << 16).view(np.float32)
            assert np.isfinite(visible).all()
            for method, values in pools(visible, spans, start, READOUTS).items():
                arrays[method][binding["slots"][sample]] = values
            cursor += tokens
        assert cursor == receipt["tokens"] and not stream.read(1)
    return tuple(torch.from_numpy(arrays[m]) for m in READOUTS)


def main():
    torch.set_num_threads(2)
    sys.path.insert(0, str(CODE))
    from common.dataset import load_split
    sys.path.insert(0, str(PROJECT / "workspaces/streaming_hd/ours/step_prefix_ordered_dual_qwen_wd1e4_20261111"))
    from model import make_model, predict
    pooling_path = CODE / "ours/step_prefix_relation_22_qwen_wd1e4_20261111/extract.py"
    pooling_data = pooling_path.read_bytes()
    tree = ast.parse(pooling_data)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pools")
    scope = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(pooling_path), "exec"), scope)
    client = boto3.Session(profile_name="runpod-qwen").client(
        "s3", endpoint_url="https://s3api-eu-ro-1.runpod.io", region_name="eu-ro-1")

    def source(key):
        with client.get_object(Bucket=BUCKET, Key=key)["Body"] as stream:
            return stream.read()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for domain in ("Logic", "Factuality"):
        freezes = {}
        for arm, (head, loss) in ARMS.items():
            rel = f"{domain}/ordered_dual/{head}/{loss}"
            root = ORIGINAL / domain / (rel if domain == "Factuality" else rel.split("/", 1)[1])
            raw = (root / "FROZEN_SELECTION.json").read_bytes()
            freeze = json.loads(raw)
            assert freeze["arm"] == rel.replace("/", ".")
            freezes[arm] = (freeze, sha(raw))
        binding = bind(domain, source, freezes["C18"][0]["data_hashes"], load_split)
        for layer in sorted({x[0]["selected"]["layer"] for x in freezes.values()}):
            arms = [arm for arm, (f, _) in freezes.items() if f["selected"]["layer"] == layer]
            if all((OUTPUT / f"{domain}.{arm}.json").exists() for arm in arms):
                continue
            print("READ_VAL", domain, layer, flush=True)
            features = pooled(client, binding, layer, scope["pools"])
            for arm in arms:
                target = OUTPUT / f"{domain}.{arm}.json"
                if target.exists():
                    continue
                head, loss = ARMS[arm]
                freeze, freeze_hash = freezes[arm]
                assert freeze["data_hashes"] == freezes["C18"][0]["data_hashes"]
                selected = freeze["selected"]
                rel = f"{domain}/ordered_dual/{head}/{loss}"
                key = ("qwen_relation/results/logic_ordered_dual/" + rel + f"/best_layer_{layer}.pt"
                       if domain == "Logic" else
                       "qwen_relation/results/factuality_c01_c20_20261112/dual/" + rel + "/best_detector.pt")
                checkpoint = source(key)
                assert sha(checkpoint) == selected["checkpoint_SHA256"]
                state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
                for name in ("layer", "epoch", "config_SHA256", "feature_SHA256"):
                    assert state[name] == selected[name]
                model = make_model(5120, head)
                model.load_state_dict(state["model"])
                probability = predict(model, features, batch=4096)
                y, c = binding["y"], binding["c"]
                prefix = probability[:, :2].sum(axis=1)
                denom = probability[:, 1:].sum(axis=1, dtype=np.float64)
                assert (denom > 0).all()
                kappa = probability[:, 1] / denom
                step_auc, step_ap = ranked_metrics(y, probability[:, 0])
                prefix_auc, prefix_ap = ranked_metrics(c, prefix)
                check = dict(step_AUROC=step_auc, prefix_AUROC=prefix_auc,
                             step_AP=step_ap, prefix_AP=prefix_ap)
                delta = {k: float(v - selected["metrics"][k]) for k, v in check.items()}
                assert max(abs(v) for v in delta.values()) <= 1e-4, (domain, arm, delta)
                scores = {}
                for label, values in (("prefix", prefix), ("kappa", kappa)):
                    scores[label] = {}
                    for name, negative in (("fresh", ~binding["prior"]), ("recovered", binding["prior"])):
                        mask = (y == 0) & ((c == 1) | negative)
                        scores[label][name] = compare(c[mask], binding["group"][mask], values[mask])
                pred = OUTPUT / f"{domain}.{arm}.val.npz"
                np.savez(pred, position_id=np.asarray(binding["keys"]), probability=probability)
                result = dict(domain=domain, arm=arm, layer=layer, epoch=selected["epoch"],
                              scores=scores, replay_val_metric_deltas=delta,
                              freeze_sha256=freeze_hash, checkpoint_sha256=sha(checkpoint),
                              prediction_sha256=sha(pred.read_bytes()), pool_code_sha256=sha(pooling_data),
                              **{k: binding[k] for k in ("labels_sha256", "offsets_sha256", "token_receipt_sha256")})
                pending = target.with_suffix(".partial")
                pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                pending.replace(target)
                print("VERIFIED", domain, arm, flush=True)


if __name__ == "__main__":
    main()
