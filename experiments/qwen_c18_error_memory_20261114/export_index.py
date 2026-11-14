import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

BASE = Path("/shared/homes/u26487468/agent_uncertainty_quantification/assets/streaming_hd_baseline_inputs/native_full/qwen3_8_27b")
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for domain in ("Logic", "Factuality"):
        for split in ("train", "val", "test"):
            raw = (BASE / domain / f"{split}.native.jsonl").read_bytes()
            rows = [dict(sample_id=r["sample_id"], token_sha=r["token_input_SHA256"],
                         origin=r["reasoning_start_token"], step_token_spans=r["step_token_spans"])
                    for r in map(json.loads, raw.splitlines())]
            data = json.dumps(dict(full_sha=hashlib.sha256(raw).hexdigest(), rows=rows),
                              separators=(",", ":")).encode()
            item = tarfile.TarInfo(f"{domain}.{split}.json")
            item.mode = 0o600
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
