"""Publish the pinned Hub labels and prepare the probe's TRAIN/VAL input contract."""
import hashlib
import json
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REV = "b45b2fd88e00869825de163bc6b7a41f31dd962b"
REPO = "pillow-streaming/streaming-hd-v1"
BUCKET = "fcsu28qkr3"
MODEL = "qwen3_8_27b"
DOMAINS = ("Logic", "Factuality")
SPLITS = ("train", "val")
FIELDS = ("sample_id", "retained_original_step_ids", "step_hallucination",
          "prefix_hallucination", "step_token_spans", "token_input_SHA256",
          "step_causal_overrides_json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(data):
    return (json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def put_once(client, key, body):
    try:
        with client.get_object(Bucket=BUCKET, Key=key)["Body"] as stream:
            assert stream.read() == body, f"Different labels already exist at {key}"
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("NoSuchKey", "404", "NotFound"):
            raise
        client.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
        with client.get_object(Bucket=BUCKET, Key=key)["Body"] as stream:
            assert digest(stream.read()) == digest(body)


def main():
    project = Path(os.environ["RHO_PROJECT"])
    out = Path(os.environ["RHO_STAGE"])
    client = boto3.Session(profile_name="runpod-qwen").client(
        "s3", endpoint_url="https://s3api-eu-ro-1.runpod.io", region_name="eu-ro-1")
    contract = {}
    for domain in DOMAINS:
        contract[domain] = {}
        for split in SPLITS:
            filename = f"data/{MODEL}/{domain}/{split}.parquet"
            path = Path(hf_hub_download(REPO, filename, repo_type="dataset", revision=REV, token=True))
            rows = pq.read_table(path, columns=list(FIELDS)).to_pylist()
            full = project / "assets/streaming_hd_baseline_inputs/native_full" / MODEL / domain / f"{split}.native.jsonl"
            native = {row["sample_id"]: row for row in map(json.loads, full.open())}
            sample_ids = [row["sample_id"] for row in rows]
            assert len(set(sample_ids)) == len(sample_ids) == len(native)
            assert set(sample_ids) == set(native)
            labels, positions = [], []
            missing = 0
            for row in rows:
                sample = row["sample_id"]
                expected = native[sample]
                assert expected["parent_reasoning_token_SHA256"] == row["token_input_SHA256"]
                assert not json.loads(row["step_causal_overrides_json"])
                steps, ys, cs, spans = (row[field] for field in FIELDS[1:5])
                assert len(steps) == len(ys) == len(cs) == len(spans)
                for idx, step in enumerate(steps):
                    assert int(ys[idx]) <= int(cs[idx])
                    assert expected["step_token_spans"][step] == spans[idx]
                    if step == 0 or ys[idx]:
                        continue
                    if idx == 0 or steps[idx - 1] != step - 1:
                        missing += 1
                    elif cs[idx - 1]:
                        positions.append(dict(sample_id=sample, span=spans[idx], label=int(cs[idx]),
                                              token_sha=expected["token_input_SHA256"]))
                labels.append({key: row[key] for key in FIELDS[:5]})
            label_bytes = encode(dict(model=MODEL, domain=domain, split=split, dataset=REPO,
                                      revision=REV, parquet_SHA256=digest(path.read_bytes()), rows=labels))
            key = f"qwen_relation/{domain}/{split}/labels.{REV[:7]}.json"
            put_once(client, key, label_bytes)
            print("LABELS_SAVED", domain, split, len(labels), len(positions), digest(label_bytes))
            contract[domain][split] = dict(full_sha=digest(full.read_bytes()), sample_ids=sample_ids,
                                           positions=positions, missing_predecessor=missing,
                                           labels_key=key, labels_sha=digest(label_bytes))
    (out / "inputs.json").write_bytes(encode(contract))


if __name__ == "__main__":
    main()
