#!/usr/bin/env bash
set -euo pipefail
umask 077
root=/workspace/qwen_relation/results/qwen_step_only_learnable_20261115/legacy_recipe
exec 9>"$root/.gpu-secondary-coordinator.lock"
flock -n 9
export STEP_GPU_OWNER=secondary STEP_THREADS=4
while read -r domain arm side; do
    export STEP_CACHE="/tmp/step_legacy_gpu_$domain"
    printf 'GPU2_SHARD_START %s %s %s %s\n' "$domain" "$arm" "$side" "$(date -u +%FT%TZ)"
    bash "$root/code/run_shard.sh" "$domain" "$arm" "$side" cuda
done < "$root/GPU2_PENDING.list"
printf 'GPU2_COMPLETE %s\n' "$(date -u +%FT%TZ)"
