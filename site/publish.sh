#!/usr/bin/env bash
# Build the weekly site and publish it to the gh-pages branch, which GitHub Pages serves.
# Usage (from the repository root, after the reports are committed to main):
#   PYTHON=/path/to/python bash site/publish.sh
# Needs the Python package `markdown`. No clone is created: the built directory is committed
# to gh-pages with a temporary index file.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
PYTHON="${PYTHON:-python3}"
OUT="$(mktemp -d)"
INDEX="$(mktemp)"
trap 'rm -rf "${OUT}" "${INDEX}"' EXIT
"${PYTHON}" site/build.py --out "${OUT}"
rm -f "${INDEX}"
GIT_INDEX_FILE="${INDEX}" git --work-tree="${OUT}" add -A .
TREE="$(GIT_INDEX_FILE="${INDEX}" git write-tree)"
PARENT="$(git ls-remote origin refs/heads/gh-pages | cut -f1)"
COMMIT="$(git commit-tree "${TREE}" ${PARENT:+-p "${PARENT}"} -m "Publish weekly site from $(git rev-parse --short HEAD)")"
git fetch -q origin gh-pages || true
git push origin "${COMMIT}:refs/heads/gh-pages"
echo "Published ${COMMIT} to gh-pages; the site updates at https://pillow-1.github.io/streaming_hd/"
