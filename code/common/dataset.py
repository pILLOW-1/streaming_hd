"""Read the agreed model/domain/split generation-and-annotation layout.

This module never downloads, resplits, rewrites labels, or derives Prefix labels.
"""
from __future__ import annotations

import json
from pathlib import Path

SPLITS = ('train', 'val', 'test')


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    ids = [row['sample_id'] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f'Empty data or duplicate sample IDs: {path}')
    return rows


def cell_root(paths: dict, model: str, domain: str) -> Path:
    """Optional cell roots support the recorded per-domain split amendments."""
    value = paths.get('dataset_roots', {}).get(model, {}).get(domain) or paths.get('dataset_root')
    if not value:
        raise ValueError('Set dataset_root (or the model/domain dataset_roots override).')
    return Path(value)


def load_split(root: str | Path, model: str, domain: str, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f'Unknown split: {split}')
    folder = Path(root) / model / domain / split
    generated = read_jsonl(folder / 'generation/records.jsonl')
    annotated = read_jsonl(folder / 'annotations/labels.jsonl')
    annotations = {row['sample_id']: row for row in annotated}
    if {row['sample_id'] for row in generated} != annotations.keys():
        raise ValueError(f'Generation/annotation coverage differs: {folder}')
    result = []
    for row in generated:
        label = annotations[row['sample_id']]
        for field in ('model', 'domain', 'question_family_id', 'canonical_question_id'):
            if row.get(field) != label.get(field):
                raise ValueError(f'Generation/annotation identity differs: {field}')
        if row['model'] != model or row['domain'] != domain:
            raise ValueError('Model/domain identity differs from directory')
        retained = row['retained_original_step_ids']
        if not retained or retained != sorted(set(retained)):
            raise ValueError('Retained Step IDs must be nonempty, ordered and unique')
        if any(type(i) is not int or not 0 <= i < len(row['original_steps']) for i in retained):
            raise ValueError('Retained Step ID is outside original_steps')
        targets = label['labels']
        for level in ('step', 'prefix'):
            if len(targets[level]) != len(retained):
                raise ValueError(f'Incomplete {level} labels: {row["sample_id"]}')
            if any(type(y) not in (int, bool) or y not in (0, 1) for y in targets[level]):
                raise ValueError(f'Non-binary {level} labels')
        if type(targets['answer']) not in (int, bool) or targets['answer'] not in (0, 1):
            raise ValueError('Missing/non-binary Answer label')
        # Normalize in memory only; source records and label files stay untouched.
        result.append(dict(row, split=split, step_hallucination=targets['step'],
                           prefix_hallucination=targets['prefix'],
                           answer_hallucination=targets['answer']))
    return result


def load_cell(paths: dict, model: str, domain: str) -> dict[str, list[dict]]:
    root = cell_root(paths, model, domain)
    rows = {split: load_split(root, model, domain, split) for split in SPLITS}
    for field in ('sample_id', 'question_family_id'):
        seen = set()
        for split, records in rows.items():
            current = {r[field] for r in records}
            if seen & current:
                raise ValueError(f'{field} crosses splits in {model}/{domain}/{split}')
            seen.update(current)
    return rows


def verify_assignment(rows: dict, model: str, domain: str, assignment: Path) -> None:
    """Required before using archived weights/results; compares ordered IDs."""
    expected = {split: [] for split in SPLITS}
    for row in read_jsonl_assignments(assignment):
        if (row['model'], row['domain']) == (model, domain):
            expected[row['split']].append(row['sample_id'])
    for split in SPLITS:
        if [r['sample_id'] for r in rows[split]] != expected[split]:
            raise ValueError(f'Data do not match archived split IDs/order: {model}/{domain}/{split}')


def read_jsonl_assignments(path: Path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def verify_archived_files(paths, model, domain, identity_file):
    """Protect archived evaluation from changed generation or annotation bytes."""
    import hashlib
    expected = [entry for entry in json.loads(Path(identity_file).read_text())
                if (entry['model'], entry['domain']) == (model, domain)]
    if {entry['split'] for entry in expected} != set(SPLITS):
        raise ValueError('Incomplete archived data identity')
    root = cell_root(paths, model, domain) / model / domain
    for entry in expected:
        for suffix, key in [('generation/records.jsonl', 'generation_SHA256'),
                            ('annotations/labels.jsonl', 'labels_SHA256')]:
            digest = hashlib.sha256()
            with (root / entry['split'] / suffix).open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
            if digest.hexdigest() != entry[key]:
                raise ValueError(f'Archived data bytes differ: {model}/{domain}/{entry["split"]}/{suffix}')
