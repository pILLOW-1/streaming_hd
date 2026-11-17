#!/bin/bash
set -euo pipefail
umask 077
base=/workspace/qwen_relation/results
root=$base/qwen_baseline_prefix_decomposition_20261116/kappa_gru_memory
prior=$base/qwen_step_only_learnable_20261115/legacy_recipe/aggr_fourhead_gated
old=$base/qwen_baseline_prefix_decomposition_20261116
control=$base/qwen_prefix_decomposition_legacy_20261117
support=$base/qwen_a2_learnable_steps_20261115/code
rank=$base/qwen_prefix_kappa_ranking_20261117/code
history=/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114
role=$1
case "$role:$RUNPOD_POD_ID" in
 lower:nwpqw43905m8nf|upper:1vbjo4pptxjeu4) ;;
 *) exit 64 ;;
esac
cd "$root"
exec 9> ".$role.dispatch.lock"
flock -n 9
trap 'rc=$?; if ((rc)); then printf "%s %s\n" "$(date -u +%FT%TZ)" "$rc" > "$role.FAILED"; fi' EXIT
sha256sum -c code/SHA256SUMS
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$role.QUEUED"
for i in $(seq 1 7200); do
 if [[ -s "$prior/$role.COMPLETE" ]]; then break; fi
 if [[ -s "$prior/$role.FAILED" ]]; then exit 75; fi
 sleep 3
done
test -s "$prior/$role.COMPLETE"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$root/code:/tmp/step_fourhead_deps:$rank:$support"
(cd "$history" && sha256sum -c --quiet SHA256SUMS)
python3 -c 'import torch,sklearn; from kappa_memory import smoke; assert torch.cuda.is_available() and torch.cuda.device_count()==1; smoke("cuda")'
read -r contract _ < <(sha256sum code/contract.json)
run() {
 local domain=$1
 shift
 python3 -u "$root/code/train.py" "$domain" streaming "$root" "$support/inputs.json" "$history" f97949f45e257e5d21200c689ce7ba791a94502cd912bd6ef8f2e2750bd47ac6 --all-positions --legacy-streaming --reuse-train-scores-from "$old" --control-root "$control" --run-contract "$contract" "$@"
}
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$role.READY"
for domain in Logic Factuality; do
 run "$domain" --layer-half "$role"
 out="$root/$domain/streaming"
 test -s "$out/$role.COMPLETE.json"
 if [[ -s "$out/lower.COMPLETE.json" && -s "$out/upper.COMPLETE.json" ]]; then
  exec 8>"$out/.test_launcher.lock"
  flock 8
  if [[ ! -s "$out/TEST_COMPLETE.json" ]]; then
   exec 7>"$out/.lock"
   flock 7
   flock -u 7
   run "$domain" --finalize
   test -s "$out/TEST_COMPLETE.json"
  fi
  flock -u 8
 fi
done
date -u +%FT%TZ > "$role.COMPLETE"
