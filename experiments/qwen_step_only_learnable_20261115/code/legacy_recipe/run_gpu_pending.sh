#!/usr/bin/env bash
set -euo pipefail
umask 077
root=/workspace/qwen_relation/results/qwen_step_only_learnable_20261115/legacy_recipe
exec 9>"$root/.gpu-coordinator.lock"
flock -n 9
export STEP_THREADS=4
run_shard() {
    local domain=$1 arm=$2 side=$3 dir
    dir="$root/$domain/$arm/$side"
    mkdir -p "$dir"
    flock "$dir/.launcher.lock" true
    export STEP_CACHE="/tmp/step_legacy_gpu_$domain"
    printf 'GPU_SHARD_START %s %s %s %s\n' "$domain" "$arm" "$side" "$(date -u +%FT%TZ)"
    bash "$root/code/run_shard.sh" "$domain" "$arm" "$side" cuda
}
for arm in repr aggr_repr; do
    for domain in Factuality Logic; do
        for side in lower upper; do run_shard "$domain" "$arm" "$side"; done
    done
done
for arm in aggr_residual aggr; do
    for domain in Factuality Logic; do
        for side in lower upper; do run_shard "$domain" "$arm" "$side"; done
    done
done
printf 'GPU_COMPLETE %s\n' "$(date -u +%FT%TZ)"
