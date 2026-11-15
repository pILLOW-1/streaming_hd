#!/usr/bin/env bash
set -euo pipefail
umask 077
domain=$1
side=$2
pod=$3
root=/workspace/qwen_relation/results/qwen_c18_error_memory_followup_20261115
index=/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114
cd "$root"
exec 9>"$root/.worker.${domain}.${side}.lock"
flock -n 9
read -r first last < <(python3.12 - "$domain" "$side" "$pod" <<'PY'
import json
import sys
domain, side, pod = sys.argv[1:]
claim = json.load(open("LAYER_WORKERS.json"))[domain]
assert side in ("lower", "upper") and claim[f"{side}_pod"] == pod
cut = claim["lower_last"]
print(*( (1, cut) if side == "lower" else (cut + 1, 64) ))
PY
)
if [[ "$side" == "lower" ]]; then
    dest="$root/$domain"
else
    dest="$root/shards/$domain/upper"
fi
export C18_FACTUALITY_OWNER="$pod"
export PYTHONPATH=/tmp/qwen_c18_cpu_deps OMP_NUM_THREADS="$(nproc)" MKL_NUM_THREADS="$(nproc)" OPENBLAS_NUM_THREADS=1
if ! python3.12 -c 'import torch,numpy,sklearn; assert (torch.__version__,numpy.__version__,sklearn.__version__)==("2.8.0+cpu","2.2.6","1.7.2")' 2>/dev/null; then
    python3.12 -m pip install --disable-pip-version-check --no-cache-dir --target "$PYTHONPATH" 'numpy==2.2.6' 'scikit-learn==1.7.2'
    python3.12 -m pip install --disable-pip-version-check --no-cache-dir --target "$PYTHONPATH" --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
fi
(cd "$index" && sha256sum -c SHA256SUMS --quiet)
cd "$root/code"
printf '%s %s owns layers %s-%s on %s\n' "$domain" "$side" "$first" "$last" "$pod"
python3.12 -u shard_train.py "$domain" "$dest" --first "$first" --last "$last" --index "$index"
python3.12 -u merge_shards.py "$root" "$domain" "$index"
