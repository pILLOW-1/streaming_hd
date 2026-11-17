#!/bin/bash
set -euo pipefail
umask 077
base=/workspace/qwen_relation/results
root=$base/qwen_step_aggr_top10_20261117
prior=$base/qwen_prefix_kappa_ranking_20261117
legacy=$base/qwen_step_only_learnable_20261115/legacy_recipe/code
history=/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114
domain=$1
case "$domain:$RUNPOD_POD_ID" in
 Logic:uonjsseqqtxjci) done_file=$prior/Logic/streaming/TEST_COMPLETE.json; failed_file=$prior/upper.FAILED ;;
 Factuality:s52mlsshua04ty) done_file=$prior/Factuality/streaming/TEST_COMPLETE.json; failed_file=$prior/lower.FAILED ;;
 *) exit 64 ;;
esac
cd "$root"
exec 9> ".$domain.dispatch.lock"
flock -n 9
trap 'rc=$?; if ((rc)); then printf "%s %s\n" "$(date -u +%FT%TZ)" "$rc" > "$domain.FAILED"; fi' EXIT
sha256sum -c code/SHA256SUMS
(cd "$legacy" && sha256sum -c "$root/code/LEGACY_SHA256SUMS")
if [[ -s "$domain/aggr_top10/TEST_COMPLETE.json" ]]; then exit 0; fi
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$domain.QUEUED"
for t in $(seq 1 7200); do
 if [[ -s "$done_file" ]]; then break; fi
 if [[ -s "$failed_file" ]]; then exit 75; fi
 sleep 2
done
test -s "$done_file"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$root/code:/tmp/cetus_prefix_k_rank_deps:$legacy"
(cd "$history" && sha256sum -c --quiet SHA256SUMS)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 "$root/code/topk.py"
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$domain.READY"
out="$root/$domain/aggr_top10"
cache="$(mktemp -d /tmp/aggr_top10.XXXXXXXX)"
trap 'rc=$?; rm -f -- "$cache"/{train.npy,train.json,val.npy,val.json,train.copying,val.copying}; rmdir "$cache" || true; if ((rc)); then printf "%s %s\n" "$(date -u +%FT%TZ)" "$rc" > "$domain.FAILED"; fi' EXIT
python3 -u "$root/code/train.py" "$domain" aggr_top10 1 32 "$out/lower" --index "$history" --cache "$cache" --device cuda
python3 -u "$root/code/train.py" "$domain" aggr_top10 33 64 "$out/upper" --index "$history" --cache "$cache" --device cuda
python3 -u -c 'from evaluate import main; from pathlib import Path; import sys; domain,out,index=sys.argv[1:]; main(domain,"aggr_top10",Path(out),Path(index),"cuda")' "$domain" "$out" "$history/$domain.test.json"
test -s "$out/TEST_COMPLETE.json"
date -u +%FT%TZ > "$domain.COMPLETE"
