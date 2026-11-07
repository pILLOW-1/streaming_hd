"""Load hash-verified, separately obtained upstream definitions without launchers."""
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def provenance(method):
    return json.loads((ROOT / 'code/baselines' / method / 'upstream/PROVENANCE.json').read_text())


def source_directory(method):
    info = provenance(method)
    root = Path(os.environ.get('STREAMING_HD_UPSTREAM_ROOT', str(ROOT / '.cache/upstream')))
    return root / method / info['commit']


def definitions(method, relative_path, names):
    """Extract named classes/functions; do not execute an upstream experiment main."""
    info = provenance(method)
    path = source_directory(method) / relative_path
    if not path.is_file():
        raise FileNotFoundError(f'Obtain the pinned {method} dependency first; see THIRD_PARTY.md: {path}')
    expected = next(x['SHA256'] for x in info['files'] if x['path'] == relative_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Upstream source hash mismatch')
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError('Upstream definitions missing')
    import math
    import numpy as np
    import torch
    namespace = dict(__name__=__name__, torch=torch, nn=torch.nn, F=torch.nn.functional, np=np, math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return tuple(namespace[name] for name in names)
