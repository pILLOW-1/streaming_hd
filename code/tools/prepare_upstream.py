#!/usr/bin/env python3
"""Obtain pinned upstream dependencies directly from their original repositories."""
import argparse
import hashlib
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from common.upstream import provenance, source_directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['hami', 'ars', 'race', 'rhd', 'llm_check'], required=True)
    p.add_argument('--accept-upstream-terms', action='store_true', help='Confirm that you have checked the author terms and are permitted to obtain/use this dependency.')
    a = p.parse_args()
    info = provenance(a.method)
    if not a.accept_upstream_terms:
        p.error('Read the original repository terms before using --accept-upstream-terms: ' + info['official_repo'])
    destination = source_directory(a.method)
    for item in info['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe upstream file path')
        url = item['source_url']
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in ('github.com', 'raw.githubusercontent.com'):
            raise ValueError('Unexpected upstream source host')
        if parsed.hostname == 'github.com':
            url = url.replace('https://github.com/', 'https://raw.githubusercontent.com/', 1).replace('/blob/', '/', 1)
        target = destination / relative
        if target.exists():
            data = target.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != item['SHA256']:
            raise ValueError('Upstream content does not match pinned revision: ' + item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as output:
                output.write(data)
    print('Verified pinned local dependency:', destination)


if __name__ == '__main__':
    main()
