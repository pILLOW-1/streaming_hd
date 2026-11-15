#!/usr/bin/env bash
set -euo pipefail
umask 077
domain=$1
arm=$2
side=$3
device=$4
root=/workspace/qwen_relation/results/qwen_step_only_learnable_20261115
index=/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114
out="$root/$domain/$arm/$side"
mkdir -p "$out"
exec 9>"$out/.launcher.lock"
flock -n 9
case "$side" in
    lower) first=1; last=32 ;;
    upper) first=33; last=64 ;;
    *) exit 64 ;;
esac
export OMP_NUM_THREADS="$(nproc)" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS="$(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$device" == cpu ]]; then
    interpreter=python3.12
    export PYTHONPATH=/tmp/a2_learner_deps
    if ! "$interpreter" -c 'import torch,numpy,sklearn' 2>/dev/null; then
        "$interpreter" -m pip install --disable-pip-version-check --no-cache-dir --target "$PYTHONPATH" 'numpy==2.2.6' 'scikit-learn==1.7.2'
        "$interpreter" -m pip install --disable-pip-version-check --no-cache-dir --target "$PYTHONPATH" --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
    fi
elif [[ "$device" == cuda ]]; then
    interpreter=python3
    export PYTHONPATH="/tmp/a2_gpu_deps:${PYTHONPATH:-}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$interpreter" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1'
    if ! "$interpreter" -c 'import sklearn' 2>/dev/null; then
        "$interpreter" -m pip install --disable-pip-version-check --no-cache-dir --target /tmp/a2_gpu_deps "numpy==2.2.6" "scikit-learn==1.7.2"
    fi
else
    exit 64
fi
(cd "$index" && sha256sum -c --quiet SHA256SUMS)
cd "$root/code"
"$interpreter" -u smoke.py "$device"
printf '%s %s layers %s-%s on %s\n' "$domain" "$arm" "$first" "$last" "$device"
"$interpreter" -u train.py "$domain" "$arm" "$first" "$last" "$out" --cache /tmp/a2_layer_cache --index "$index" --device "$device"
"$interpreter" -u evaluate.py "$domain" "$arm" "$root/$domain/$arm" "$index/$domain.test.json" "$device"
