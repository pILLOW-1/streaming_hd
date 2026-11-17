#!/bin/bash
set -euo pipefail
umask 077
base=/workspace/qwen_relation/results
root=$base/qwen_step_only_learnable_20261115/legacy_recipe/aggr_full20
legacy=$base/qwen_step_only_learnable_20261115/legacy_recipe/code
prior=$base/qwen_baseline_prefix_decomposition_20261116/kappa_gru_memory
index=/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114
arm=aggr_full20
case "$1:$RUNPOD_POD_ID" in
 lower:nwpqw43905m8nf) side=lower; a=1; b=32 ;;
 upper:1vbjo4pptxjeu4) side=upper; a=33; b=64 ;;
 *) exit 64 ;;
esac
cd "$root"
exec 9> ".$side.dispatch.lock"
flock -n 9
trap 'rc=$?; if ((rc)); then printf "%s %s\n" "$(date -u +%FT%TZ)" "$rc" > "$side.FAILED"; fi' EXIT
sha256sum -c code/SHA256SUMS
(cd "$legacy" && sha256sum -c "$root/code/LEGACY_SHA256SUMS")
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$side.QUEUED"
for i in $(seq 1 28800); do
 if [[ -s "$prior/$side.COMPLETE" ]]; then break; fi
 if [[ -s "$prior/$side.FAILED" ]]; then exit 75; fi
 sleep 3
done
test -s "$prior/$side.COMPLETE"
cache="$(mktemp -d /tmp/step_full20.XXXXXXXX)"
trap 'rc=$?; rm -f -- "$cache"/{train.npy,train.json,val.npy,val.json,train.copying,val.copying}; rmdir "$cache" || true; if ((rc)); then printf "%s %s\n" "$(date -u +%FT%TZ)" "$rc" > "$side.FAILED"; fi' EXIT
export PYTHONPATH="$root/code:/tmp/step_fourhead_deps:$legacy"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$index" && sha256sum -c --quiet SHA256SUMS)
python3 -c 'import torch,sklearn; assert torch.cuda.is_available() and torch.cuda.device_count()==1'
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$side.READY"
for domain in Logic Factuality; do
 out="$root/$domain/$arm"
 python3 -u code/worker.py fit "$domain" "$arm" "$a" "$b" "$out/$side" --cache "$cache" --index "$index" --device cuda
 test -s "$out/$side/SHARD_COMPLETE.json"
 if [[ -s "$out/lower/SHARD_COMPLETE.json" && -s "$out/upper/SHARD_COMPLETE.json" ]]; then
  exec 8>"$out/.test_launcher.lock"
  flock 8
  if [[ ! -s "$out/TEST_COMPLETE.json" ]]; then
   python3 -u code/worker.py test "$domain" "$out" "$index/$domain.test.json"
   test -s "$out/TEST_COMPLETE.json"
  fi
  flock -u 8
 fi
done
date -u +%FT%TZ > "$side.COMPLETE"
