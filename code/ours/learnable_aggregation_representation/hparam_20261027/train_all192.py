#!/usr/bin/env python3
"""One continuous 192-candidate family, six independent process workers/layer.

Every worker owns its model, BN, Adam, scheduler, RNG, order and full checkpoint.
Global selection never runs until all192 fullVAL receipts are verified.
"""
import copy
import concurrent.futures
import fcntl
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from formal_common import N, sha, save, identity, lease_gpu
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recovered_20261027"))
from shared_representation import SharedRepresentationStreaming

STOP = False
def stop(_sig, _frame):
    global STOP
    STOP = True


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fingerprint(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                  for v in model.state_dict().values())).hexdigest()


def open_data(info, metadata):
    meta = torch.load(metadata, map_location='cpu', weights_only=False)
    if info['format'] == 'native_torch':
        data = torch.load(info['path'], map_location='cpu', mmap=True, weights_only=False)
        tokens = data['tokens']
        assert torch.equal(data['labels'], meta['labels']) and torch.equal(data['offsets'], meta['offsets'])
    else:
        mapped = np.memmap(info['path'], dtype=np.uint16, mode='c', shape=(meta['tokens'], 4096))
        tokens = torch.from_numpy(mapped).view(torch.bfloat16)
    return dict(meta, tokens=tokens)


def predict(model, data):
    model.eval()
    scores, losses = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(data['labels']), 256):
            ix = np.arange(start, min(start + 256, len(data['labels'])))
            logits = model.forward_bags(data['tokens'], data['offsets'], ix, 'cuda')[0]
            assert torch.isfinite(logits).all()
            target = data['labels'][ix].float().cuda()
            losses += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='sum'))
            scores.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(scores), losses / len(data['labels'])


def worker(candidate, inputs, barrier):
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    contract = identity()
    cid = candidate['candidate_id']
    directory = N / 'results' / cid
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = N / 'checkpoints' / cid
    checkpoint.mkdir(parents=True, exist_ok=True)
    train, val = [open_data(inputs[s], N / 'metadata' / f'{s}.pt') for s in ['train', 'val']]
    model = SharedRepresentationStreaming(4096).cuda()
    expected_init = json.loads((N / 'audits/SOURCE_REVIEW.json').read_text())['architecture']['six_initial_state_SHA'][0]
    assert fingerprint(model) == expected_init
    initial = cpu(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=candidate['lr'], weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=20, min_lr=1e-6)
    generator = torch.Generator().manual_seed(42)
    state = dict(epoch=1, cursor=0, order=None, train_sum=0.0, seen=0, without=0,
                 best_auc=-1.0, best_ap=-1.0, best_epoch=0, trace=[], updates=0)
    resume = checkpoint / 'last.pt'
    torch.cuda.reset_peak_memory_stats()
    # A real simultaneous worst256 update and complete state restore gate. These
    # updates are discarded before formal training and are charged as gates.
    barrier.wait(timeout=300)
    lengths = np.diff(train['offsets'].numpy())
    worst = np.argsort(lengths, kind='stable')[-256:]
    calls = []
    handle = model.mapper[1].register_forward_hook(lambda module, args, output: calls.append(len(args[0])))
    def update(ix):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model.forward_bags(train['tokens'], train['offsets'], ix, 'cuda')[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, train['labels'][ix].float().cuda())
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        opt.step()
        return float(loss.detach())
    update(worst)
    handle.remove()
    assert calls == [int(lengths[worst].sum())]
    restore = dict(model=cpu(model.state_dict()), optimizer=cpu(opt.state_dict()), scheduler=scheduler.state_dict(),
                   CPU=torch.get_rng_state(), CUDA=torch.cuda.get_rng_state(), numpy=np.random.get_state(),
                   python=random.getstate(), shuffle=generator.get_state())
    atomic_torch(checkpoint / 'concurrent_gate.pt', restore)
    ix = np.arange(candidate['batch'])
    update(ix)
    expected = cpu(model.state_dict())
    expected_opt = cpu(opt.state_dict())
    saved = torch.load(checkpoint / 'concurrent_gate.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['CPU'])
    torch.cuda.set_rng_state(saved['CUDA'])
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    generator.set_state(saved['shuffle'])
    update(ix)
    assert all(torch.equal(v.cpu(), expected[k]) for k, v in model.state_dict().items())
    for key, values in opt.state_dict()['state'].items():
        for name, value in values.items():
            assert torch.equal(value.cpu(), expected_opt['state'][key][name])
    save(directory / 'CONCURRENT_GATE.json', dict(status='PASS_REAL_CONCURRENT256_AND_FULL_RESTORE',
         initial_SHA=expected_init, worst256_tokens=int(lengths[worst].sum()), BN_calls=calls,
         GPU_private_peak_bytes=torch.cuda.max_memory_allocated(), next_update_error=0.0,
         job=os.environ['PBS_JOBID']))
    del restore, expected, expected_opt, saved
    model.load_state_dict(initial)
    opt = torch.optim.Adam(model.parameters(), lr=candidate['lr'], weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=20, min_lr=1e-6)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    generator.manual_seed(42)
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert saved['candidate'] == candidate and saved['contract_SHA'] == sha(N / 'configs/CONTRACT.json')
        assert saved['source_inputs'] == inputs
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['CPU'])
        torch.cuda.set_rng_state(saved['CUDA'])
        random.setstate(saved['python'])
        np.random.set_state(saved['numpy'])
        state = saved['training']
        del saved
    barrier.wait(timeout=300)
    barrier.wait(timeout=300)  # Parent records actual six-process residency before fits.
    if (directory / 'ACCEPTED.json').exists():
        return
    begin = time.monotonic()
    def commit():
        atomic_torch(resume, dict(candidate=candidate, contract_SHA=sha(N / 'configs/CONTRACT.json'),
            trainer_SHA=sha(Path(__file__)), architecture_SHA=sha(N / 'source/shared_representation.py'),
            source_inputs=inputs, initial_SHA=expected_init, model=cpu(model.state_dict()),
            optimizer=cpu(opt.state_dict()), scheduler=scheduler.state_dict(), generator=generator.get_state(),
            CPU=torch.get_rng_state(), CUDA=torch.cuda.get_rng_state(), python=random.getstate(),
            numpy=np.random.get_state(), training=state))
        save(directory / 'LIVE.json', dict(candidate=cid, epoch=state['epoch'], cursor=state['cursor'],
             updates=state['updates'], completed_epochs=len(state['trace']),
             elapsed_current_worker_seconds=time.monotonic() - begin, job=os.environ['PBS_JOBID']))
    while state['epoch'] <= 20 and state['without'] < 5:
        if state['order'] is None:
            state['order'] = torch.randperm(len(train['labels']), generator=generator).numpy()
        model.train()
        for start in range(state['cursor'], len(state['order']), candidate['batch']):
            ix = state['order'][start:start + candidate['batch']]
            loss = update(ix)
            state['train_sum'] += loss * len(ix)
            state['seen'] += len(ix)
            state['updates'] += 1
            state['cursor'] = start + len(ix)
            if state['updates'] % 50 == 0 or STOP:
                commit()
            if STOP:
                return
        assert state['seen'] == 59847
        trainloss = state['train_sum'] / state['seen']
        scheduler.step(trainloss)
        score, valloss = predict(model, val)
        y = val['labels'].numpy().astype(np.int64)
        auc, ap = float(roc_auc_score(y, score)), float(average_precision_score(y, score))
        assert np.isfinite(score).all()
        row = dict(epoch=state['epoch'], train_loss=trainloss, val_loss=valloss, val_auroc=auc,
                   val_auprc=ap, lr=opt.param_groups[0]['lr'], updates=state['updates'],
                   elapsed_seconds=time.monotonic() - begin,
                   parameter_norm=float(sum(p.detach().square().sum() for p in model.parameters()).sqrt()),
                   BN_batches=int(model.mapper[1].num_batches_tracked),
                   gradient_norm=float(sum(p.grad.detach().square().sum() for p in model.parameters()).sqrt()),
                   BN_running_mean_norm=float(model.mapper[1].running_mean.norm()),
                   BN_running_var_norm=float(model.mapper[1].running_var.norm()))
        state['trace'].append(row)
        # Early stopping follows AUROC only; AP only resolves checkpoint ties.
        state['without'] = 0 if auc > state['best_auc'] else state['without'] + 1
        if (auc, ap) > (state['best_auc'], state['best_ap']):
            state.update(best_auc=auc, best_ap=ap, best_epoch=state['epoch'])
            atomic_torch(checkpoint / 'best.pt', dict(state_dict=cpu(model.state_dict()), candidate=candidate,
                trainer_SHA=sha(Path(__file__)), architecture_SHA=sha(N / 'source/shared_representation.py'),
                source_inputs=inputs, initial_SHA=expected_init, best_epoch=state['epoch'],
                contract_SHA=sha(N / 'configs/CONTRACT.json')))
            temporary = directory / 'val_predictions.tmp.npz'
            np.savez_compressed(temporary, scores=score, labels=y, sample_ids=np.asarray(val['samples']))
            temporary.replace(directory / 'val_predictions.npz')
        save(directory / 'TRACE.json', state['trace'])
        print(json.dumps(dict(event='CANDIDATE_EPOCH', candidate=cid, **row)), flush=True)
        state.update(epoch=state['epoch'] + 1, cursor=0, order=None, train_sum=0.0, seen=0)
        commit()
    prediction = directory / 'val_predictions.npz'
    saved = np.load(prediction)
    assert len(saved['scores']) == 20245 and np.array_equal(saved['labels'], val['labels'].numpy())
    assert abs(roc_auc_score(saved['labels'], saved['scores']) - state['best_auc']) < 1e-12
    receipt = dict(status='PASS_FULL_VAL_CANDIDATE', candidate=candidate,
                   initial_SHA=expected_init, best_epoch=state['best_epoch'], epochs_spent=len(state['trace']),
                   val_auroc=state['best_auc'], val_auprc=state['best_ap'],
                   checkpoint=str(checkpoint / 'best.pt'), checkpoint_SHA=sha(checkpoint / 'best.pt'),
                   predictions=str(prediction), predictions_SHA=sha(prediction),
                   metadata_SHA=sha(N / 'metadata/val.pt'), TEST_evaluations=0,
                   contract_SHA=sha(N / 'configs/CONTRACT.json'), job=os.environ['PBS_JOBID'])
    save(directory / 'ACCEPTED.json', receipt)


def staged_layer(layer, work):
    native = json.loads((N / 'audits/FORMAL_INPUT_ACCEPTANCE.json').read_text())['sources']
    if layer == 16:
        sources = {split: dict(s, format='native_torch') for split, s in native.items()}
    else:
        complete = N / 'audits/ALL32_SOURCE_COMPLETE.json'
        if not complete.exists():
            return None
        family = json.loads(complete.read_text())
        assert family['status'] == 'PASS_ALL32_EXACT_NATIVE_TRAINVAL_SOURCES'
        sources = {split: family['sources'][split][str(layer)] for split in ['train', 'val']}
    out = {}
    for split, record in sources.items():
        path = Path(record['path'])
        target = work / 'features' / f'{split}_layer{layer:02d}{path.suffix}'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != path.stat().st_size or sha(target) != record['SHA']:
            temporary = target.with_suffix(target.suffix + '.copying')
            shutil.copyfile(path, temporary)
            assert sha(temporary) == record['SHA']
            temporary.replace(target)
        out[split] = dict(record, path=str(target), canonical_path=str(path))
    return out


def main():
    contract = identity()
    owner = (N / 'audits/train192.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = lease_gpu(20 * 1024**3)
    work = Path(os.environ['LHP_SCRATCH'])
    live = []
    def request_checkpoint(sig, frame):
        stop(sig, frame)
        for process in live:
            if process.is_alive():
                os.kill(process.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_checkpoint)
    signal.signal(signal.SIGALRM, request_checkpoint)
    allocation = json.loads(subprocess.check_output(
        ['qstat', '-xf', '-F', 'json', os.environ['PBS_JOBID']], text=True))['Jobs'][os.environ['PBS_JOBID']]
    def seconds(text):
        h, m, s = map(int, text.split(':'))
        return 3600 * h + 60 * m + s
    remaining = seconds(allocation['Resource_List']['walltime']) - seconds(
        allocation.get('resources_used', {}).get('walltime', '00:00:00'))
    signal.alarm(max(1, remaining - 600))
    plan = json.loads((N / 'manifests/CANDIDATE_PLAN.json').read_text())['candidates']
    # Resource-aware pipeline: reuse16 immediately, then other fixed layers.
    for layer in [16] + [x for x in range(1, 33) if x != 16]:
        ownership_path = N / 'configs/CANDIDATE_EXTERNAL_OWNERSHIP.json'
        if ownership_path.exists():
            external = json.loads(ownership_path.read_text())
            assert external['status'] in ['EXTERNAL_AUTODL_OWNED_PENDING_LOCAL_SETUP', 'EXTERNAL_AUTODL_RUNNING', 'EXTERNAL_AUTODL_VAL_COMPLETE']
            assert external['contract_SHA'] == sha(N / 'configs/CONTRACT.json')
            if layer in external['external_layers']:
                continue
        group = [c for c in plan if c['layer'] == layer]
        assert len(group) == 6
        # Strip preparation-only status fields from scientific worker identity.
        group = [{k: c[k] for k in ['candidate_id', 'layer', 'lr', 'batch', 'seed']} for c in group]
        if all((N / 'results' / c['candidate_id'] / 'ACCEPTED.json').exists() for c in group):
            continue
        group = [c for c in group if not (N / 'results' / c['candidate_id'] / 'ACCEPTED.json').exists()]
        inputs = staged_layer(layer, work)
        if inputs is None:
            save(N / 'audits/TRAIN_EXIT.json', dict(status='WAITING_FULL32_SOURCE', job=os.environ['PBS_JOBID']))
            return
        context = mp.get_context('spawn')
        barrier = context.Barrier(len(group) + 1)
        processes = [context.Process(target=worker, args=(c, inputs, barrier)) for c in group]
        live[:] = processes
        for process in processes:
            process.start()
        try:
            barrier.wait(timeout=300)
            barrier.wait(timeout=300)
            actual = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True)
            gates = [json.loads((N / 'results' / c['candidate_id'] / 'CONCURRENT_GATE.json').read_text()) for c in group]
            assert all(g['next_update_error'] == 0 and g['worst256_tokens'] == 23356 for g in gates)
            save(N / 'audits' / f'CONCURRENT_LAYER{layer:02d}_GATE.json',
                 dict(status='PASS_INDEPENDENT_CONCURRENT256_WORKERS', workers=len(group), actual_GPU_residency=actual,
                      gates=gates, job=os.environ['PBS_JOBID']))
            barrier.wait(timeout=300)
            for process in processes:
                process.join()
            assert all(p.exitcode == 0 for p in processes), [(p.pid, p.exitcode) for p in processes]
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=60)
            raise
        if STOP:
            save(N / 'audits/TRAIN_EXIT.json', dict(status='CONTINUE_FULL_STATE', job=os.environ['PBS_JOBID']))
            return
        assert all((N / 'results' / c['candidate_id'] / 'ACCEPTED.json').exists() for c in group)
        receipts = list((N / 'results').glob('*/ACCEPTED.json'))
        save(N / 'reports/TRAIN_PROGRESS.json', dict(completed_candidates=len(receipts), total=192,
             completed_layer=layer, job=os.environ['PBS_JOBID'], TEST_evaluations=0))
        # Only these two owned staged copies are disposable; canonical inputs stay.
        for record in inputs.values():
            path = Path(record['path'])
            assert path.parent == work / 'features' and path != Path(record['canonical_path'])
            path.unlink(missing_ok=True)  # Staged inputs are disposable; prior removal is harmless.
    if len(list((N / 'results').glob('*/ACCEPTED.json'))) != 192:
        save(N / 'audits/TRAIN_EXIT.json', dict(status='WAITING_EXTERNAL_FULL_VAL', job=os.environ['PBS_JOBID']))
        return
    from select_and_test import freeze_selection
    freeze_selection()
    save(N / 'audits/TRAIN_EXIT.json', dict(status='ALL192_VAL_SELECTED', job=os.environ['PBS_JOBID']))


if __name__ == '__main__':
    main()
