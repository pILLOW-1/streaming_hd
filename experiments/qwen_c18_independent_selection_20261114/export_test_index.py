"""Export minimal frozen TEST span indices on Cetus, without text, token IDs or labels."""
import hashlib
import json
import sys
import tarfile
import io
from pathlib import Path

ROOT = Path("/shared/homes/u26487468/agent_uncertainty_quantification")
BASE = ROOT / "assets/streaming_hd_baseline_inputs/native_full/qwen3_8_27b"
DOMAINS = {
    "Logic": "057de422f6542c8fcbb4552e3558e6681e7da4fac8729a1458c70a5b6074f5ae",
    "Factuality": "68c6b6b73344216825ceb43318ac80337ce72fb29782957b3f1dd2a063498f51",
}


with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
    for domain, expected in DOMAINS.items():
        raw = (BASE / domain / "test.native.jsonl").read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == expected
        rows = [dict(sample_id=r["sample_id"], token_sha=r["token_input_SHA256"],
                     origin=r["reasoning_start_token"], step_token_spans=r["step_token_spans"])
                for r in map(json.loads, raw.splitlines())]
        data = json.dumps(dict(full_sha=sha, rows=rows), separators=(",", ":")).encode()
        item = tarfile.TarInfo(f"{domain}.index.json")
        item.mode = 0o600
        item.size = len(data)
        tar.addfile(item, io.BytesIO(data))
