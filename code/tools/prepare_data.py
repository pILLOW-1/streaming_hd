#!/usr/bin/env python3
"""Validate local data in the agreed layout; do not download or change splits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from common.dataset import load_cell, verify_assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--model', help='Validate one generator, or all configured models.')
    parser.add_argument('--domain', help='Validate one domain, or all five.')
    parser.add_argument('--check-archived-assignment', action='store_true')
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text())
    models = [args.model] if args.model else list(paths['model_paths'])
    domains = [args.domain] if args.domain else ['Math', 'Logic', 'Science', 'Multi-hop', 'Factuality']
    report = []
    for model in models:
        for domain in domains:
            rows = load_cell(paths, model, domain)
            if args.check_archived_assignment:
                verify_assignment(rows, model, domain, ROOT / 'experiments/streaming_622_p70/split_assignments.jsonl')
            report.append(dict(model=model, domain=domain, counts={s: len(r) for s, r in rows.items()}))
    print(json.dumps(dict(status='PASS', cells=report), indent=2))


if __name__ == '__main__':
    main()
