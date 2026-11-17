#!/bin/bash
set -euo pipefail
umask 077
base=/workspace/qwen_relation/results
root=$base/qwen_prefix_kappa_ranking_20261117
old=$base/qwen_baseline_prefix_decomposition_20261116
control=$base/qwen_prefix_decomposition_legacy_20261117
support=$base/qwen_a2_learnable_steps_20261115/code
history=/workspace/qwen_relation/indexes/qwen_c18_frozen_history_20261114
role=$1
cd "$root"
exec 9> ".$role.dispatch.lock"
flock -n 9
trap 'rc=$?; if ((rc)); then printf "%s %s\n" "$(date -u +%FT%TZ)" "$rc" > "$role.FAILED"; fi' EXIT
sha256sum -c code/SHA256SUMS
python3 -c 'import json,sys,os; assert json.load(open("OWNERS.json"))[sys.argv[1]] == os.environ["RUNPOD_POD_ID"]' "$role"
test -s "$old/Logic/streaming/train_step.npz"
test -s "$old/Factuality/streaming/train_step.npz"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$root/code:/tmp/cetus_prefix_k_rank_deps:$support"
if ! python3 -c 'import sklearn' 2>/dev/null; then
 python3 -m pip install --disable-pip-version-check --no-cache-dir --target /tmp/cetus_prefix_k_rank_deps 'numpy==2.2.6' 'scikit-learn==1.7.2'
fi
python3 - <<'PY'
import numpy as np
import torch
from ranking import RecoveryPairs,pair_loss
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p=RecoveryPairs(["a:0","a:1","a:2","a:3","b:0","b:1","b:2"], [1,0,0,0,1,0,0], [1,1,0,1,1,0,1])
x=torch.tensor([[0.],[2.],[-2.],[2.],[0.],[-2.],[2.]],device="cuda")
m=torch.nn.Linear(1,1,bias=False,device="cuda")
m.weight.data.fill_(1)
loss=pair_loss(m,x,p.draw(np.random.default_rng(2),"cuda"))
torch.testing.assert_close(loss,torch.nn.functional.softplus(torch.tensor(-4.,device="cuda")))
loss.backward()
assert torch.isfinite(m.weight.grad).all()
assert p.evaluate(x.flatten().cpu().numpy()) == dict(AUROC=1.0,trajectories=2,pairs=3)
print("CUDA_PAIR_GATE_PASS",flush=True)
PY
read -r contract _ < <(sha256sum code/contract.json)
run() {
 local domain=$1
 shift
 python3 -u "$root/code/train.py" "$domain" streaming "$root" "$support/inputs.json" "$history"   f97949f45e257e5d21200c689ce7ba791a94502cd912bd6ef8f2e2750bd47ac6   --all-positions --legacy-streaming --reuse-train-scores-from "$old" --control-root "$control" --run-contract "$contract" "$@"
}
printf '%s %s\n' "$(date -u +%FT%TZ)" "$RUNPOD_POD_ID" > "$role.READY"
case "$role" in
 lower)
  run Logic --layer-half lower
  run Factuality
  test -s Factuality/streaming/TEST_COMPLETE.json
  date -u +%FT%TZ > Factuality.COMPLETE
  ;;
 upper)
  run Logic --layer-half upper
  for i in $(seq 1 5400); do
   if [[ -s Logic/streaming/lower.COMPLETE.json ]] && flock -n Logic/streaming/.lock true; then break; fi
   if [[ -s lower.FAILED ]] && [[ ! -s Logic/streaming/lower.COMPLETE.json ]]; then exit 75; fi
   sleep 2
  done
  test -s Logic/streaming/lower.COMPLETE.json
  run Logic --finalize
  test -s Logic/streaming/TEST_COMPLETE.json
  date -u +%FT%TZ > Logic.COMPLETE
  ;;
 *) exit 64 ;;
esac
