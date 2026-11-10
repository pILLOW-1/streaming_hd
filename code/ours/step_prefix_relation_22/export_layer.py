"""One bounded CPU-only source chunk exports one pinned layer for one consumer."""
import json
import os
import shutil
import sys
from pathlib import Path
from relation_common import source, sha, save
import hashlib

domain, readout, layer, target = sys.argv[1:]
receipt, name, receipt_hash = source(domain, readout)
from relation_common import STATE
binding = json.loads((STATE / (domain + '.json')).read_text())
positions = json.loads((Path(receipt['feature_root']) / 'POSITIONS.json').read_text())
assert hashlib.sha256(json.dumps(positions).encode()).hexdigest() == binding['position_hash']
assert os.environ.get('PBS_JOBID')
assert os.uname().nodename.split('.')[0] == receipt['host'].split('.')[0]
key = 'layer_' + layer + '/' + name + '.npy'
pin = receipt['files'][key]
src = Path(receipt['feature_root']) / key
target = Path(target)
assert target.parent.name == os.environ['PBS_JOBID'], 'Private transport must belong to this PBS job'
target.parent.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(target.parent).free > pin['bytes'] + (32 << 30)
temp = target.with_suffix('.tmp')
with src.open('rb') as f, temp.open('wb') as out:
    while True:
        b = f.read(8 << 20)
        if not b: break
        out.write(b)
        out.flush(); os.fdatasync(out.fileno())
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(f.fileno(), 0, f.tell(), os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(out.fileno(), 0, out.tell(), os.POSIX_FADV_DONTNEED)
assert sha(temp) == pin['SHA256']
temp.replace(target)
save(target.with_suffix('.json'), dict(source_manifest_SHA256=receipt_hash, file=key, **pin))
