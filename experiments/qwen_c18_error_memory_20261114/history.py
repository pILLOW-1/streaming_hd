"""Read retained C18 positions and all causally visible historical steps."""
import json
from pathlib import Path

import numpy as np
import torch

from pooling import pools
from run import Split, digest


class HistorySplit(Split):
    def __init__(self, domain, split, source, index_path, labels=None):
        index_bytes = Path(index_path).read_bytes()
        self.index = json.loads(index_bytes)
        super().__init__(domain, split, source, rows=labels)
        assert self.index["full_sha"] == self.receipt["full_native_SHA256"]
        self.pins["history_index"] = digest(index_bytes)
        native = {r["sample_id"]: r for r in self.index["rows"]}
        offsets = {r["sample_id"]: r for r in json.loads((self.root / "TOKEN_OFFSETS.json").read_bytes())}
        assert len(native) == len(offsets) and set(native) == set(offsets)
        self.history = []
        starts, positions = [], np.empty(len(self.keys), np.int64)
        cursor = 0
        for offset, origin, spans, slots in self.rows:
            ids = [key.rsplit(":", 1) for key in self.keys[slots]]
            sample = ids[0][0]
            row = native[sample]
            entry = offsets[sample]
            assert offset == entry["offset"] and row["token_sha"] == entry["token_SHA256"]
            assert origin == row["origin"]
            steps = [int(step) for s, step in ids if s == sample]
            assert len(steps) == len(ids) and steps == sorted(set(steps))
            all_spans = row["step_token_spans"][:steps[-1] + 1]
            assert [all_spans[i] for i in steps] == list(spans)
            assert all(origin <= a < b <= entry["spans"][0][1] for a, b in all_spans)
            assert all(a[1] <= b[0] for a, b in zip(all_spans, all_spans[1:]))
            history = slice(cursor, cursor + len(all_spans))
            self.history.append((offset, origin, all_spans, history, slots, steps))
            starts.extend([cursor] * len(all_spans))
            positions[slots] = cursor + np.asarray(steps)
            cursor = history.stop
        self.starts = torch.tensor(starts, dtype=torch.long)
        self.positions = torch.from_numpy(positions)

    def layer(self, layer, device):
        name = f"layer_{layer:02d}/reasoning_answer.bf16.npy"
        path = self.root / name
        pin = self.receipt["files"][name]
        assert pin["layer"] == layer and path.stat().st_size == pin["bytes"]
        raw = np.load(path, allow_pickle=False, mmap_mode="r")
        assert raw.shape == (self.receipt["tokens"], 5120) and raw.dtype == np.dtype("<u2")
        all_steps = np.empty((len(self.starts), 5120), np.float32)
        prefix = np.empty((len(self.positions), 5120), np.float32)
        for offset, origin, spans, history, slots, steps in self.history:
            length = spans[-1][1] - origin
            visible = (raw[offset:offset + length].astype(np.uint32) << 16).view(np.float32)
            assert np.isfinite(visible).all()
            all_steps[history] = pools(visible, spans, origin, ("step_exp",))["step_exp"]
            prefix[slots] = pools(visible, [spans[i] for i in steps], origin, ("prefix_exp",))["prefix_exp"]
        return tuple(x.to(device) for x in (torch.from_numpy(all_steps), torch.from_numpy(prefix),
                                             self.positions, self.starts))
