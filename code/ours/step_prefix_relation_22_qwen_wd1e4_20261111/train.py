"""One arm processes all64 layers continuously with independent full-state training."""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from relation_common import *
from model import make_model, objective, predict


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def val_stats(y, c, p):
    sy, sc = p[:, 0], p[:, :2].sum(1)
    ay, ac = roc_auc_score(y, sy), roc_auc_score(c, sc)
    py, pc = average_precision_score(y, sy), average_precision_score(c, sc)
    bce = 0.5 * (log_loss(y, np.clip(sy, 1e-15, 1-1e-15), labels=[0, 1]) +
                 log_loss(c, np.clip(sc, 1e-15, 1-1e-15), labels=[0, 1]))
    return dict(mean_AUROC=float((ay+ac)/2), mean_AP=float((py+pc)/2), mean_BCE=float(bce),
                step_AUROC=float(ay), prefix_AUROC=float(ac), step_AP=float(py), prefix_AP=float(pc))


def rank(m):
    return m['mean_AUROC'], m['mean_AP'], -m['mean_BCE']


def atomic_torch(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def main(arm):
    assert torch.cuda.is_available() and os.environ.get('PBS_JOBID')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cfg = config(); domain, readout, head, loss = arm.split('.')
    out, checkpoints = arm_paths(arm); out.mkdir(parents=True, exist_ok=True)
    if (out / 'FROZEN_SELECTION.json').exists(): return 0
    contract_hash = sha(WORK / 'config.json')
    data = dataset(domain, ('train', 'val'))
    source_receipt, source_name, receipt_hash = source(domain, readout)
    identity = source_receipt['identity']
    keys = data['train']['keys'] + data['val']['keys']
    assert identity['ranges']['train'] == [0, len(data['train']['keys'])]
    assert identity['ranges']['val'] == [len(data['train']['keys']), len(keys)]
    # CPU metadata exported by setup binds every position, native file and label hash.
    binding = json.loads((STATE / (domain + '.json')).read_text())
    assert binding['data_hashes'] == data_hashes(domain)
    assert binding['position_hash'] == hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    assert identity['native_files'] == binding['native_files']
    assert identity['positions'] == len(keys)
    assert len(keys) == len(set(keys))
    save(out / 'CONTRACT.json', dict(arm=arm, config_SHA256=contract_hash, data=binding,
                                    source_manifest_SHA256=receipt_hash, candidate_layers=cfg['layers'], feature_L2=False))
    n = len(data['train']['keys']); y = torch.from_numpy(data['train']['y']).cuda(); c = torch.from_numpy(data['train']['c']).cuda()
    root = SCRATCH / 'fits' / arm; root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); deadline = start + float(os.environ.get('EXECUTION_SECONDS', '19800'))
    for layer in cfg['layers']:
        report = out / ('layer_' + str(layer) + '.json')
        if report.exists():
            old = json.loads(report.read_text()); assert old['config_SHA256'] == contract_hash
            continue
        if time.monotonic() + 180 > deadline: return 75
        pin = source_receipt['files'][f'layer_{layer}/{source_name}.npy']
        local = root / (pin['SHA256'] + '.npy')
        if not local.exists():
            transport = PROJECT / 'assets/step_prefix_relation_transport' / os.environ['PBS_JOBID'] / (pin['SHA256'] + '.npy')
            subprocess.run(['/opt/pbs/bin/pbsdsh', '-n', '1', '--', '/usr/bin/python3.12',
                            str(WORK / 'export_layer.py'), domain, readout, str(layer), str(transport)], check=True)
            assert sha(transport) == pin['SHA256']
            shutil.copyfile(transport, local.with_suffix('.tmp'))
            assert sha(local.with_suffix('.tmp')) == pin['SHA256']
            local.with_suffix('.tmp').replace(local)
            # Only this job's private transport copy, after destination hash verification.
            transport.unlink()
        assert sha(local) == pin['SHA256']
        x = np.load(local, mmap_mode='r', allow_pickle=False)
        assert x.shape == (len(keys), 5120) and x.dtype == np.float32 and np.isfinite(x).all()
        tx = torch.from_numpy(np.array(x[:n], copy=True)).cuda()
        vx = torch.from_numpy(np.array(x[n:], copy=True)).cuda()
        del x
        seed_all(cfg['seed']); net = make_model(5120, head).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        progress = checkpoints / ('progress_layer_' + str(layer) + '.pt')
        bestpath = checkpoints / ('best_layer_' + str(layer) + '.pt')
        begin, stale, best, completed_epochs = 0, 0, None, 0
        resume_order, resume_offset = None, 0
        generator = torch.Generator(device='cpu')
        generator.manual_seed(cfg['seed'])
        if progress.exists():
            state = torch.load(progress, map_location='cpu', weights_only=False)
            assert state['config_SHA256'] == contract_hash and state['feature_SHA256'] == pin['SHA256']
            net.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            for group in optimizer.state.values():
                for k, v in group.items():
                    if torch.is_tensor(v): group[k] = v.cuda()
            begin = state['epoch']; stale = state['stale']; best = state['best']
            completed_epochs = begin
            generator.set_state(state['permutation_rng']); torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
            resume_order=state.get('batch_order');resume_offset=state.get('batch_offset',0)
        for epoch in range(begin, cfg['max_epochs']):
            if stale >= cfg['patience']: break
            if time.monotonic() + 120 > deadline: return 75
            net.train(); order = resume_order if resume_order is not None else torch.randperm(n, generator=generator)
            for offset in range(resume_offset, n, cfg['batch_size']):
                if time.monotonic()+60>deadline:
                    atomic_torch(progress,dict(model=net.state_dict(),optimizer=optimizer.state_dict(),
                        epoch=epoch,stale=stale,best=best,config_SHA256=contract_hash,feature_SHA256=pin['SHA256'],
                        permutation_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        batch_order=order,batch_offset=offset))
                    return 75
                batch = order[offset:offset+cfg['batch_size']].cuda()
                optimizer.zero_grad(set_to_none=True)
                value = objective(net(tx[batch]), y[batch], c[batch], loss)
                assert torch.isfinite(value)
                value.backward(); optimizer.step()
            p = predict(net, vx)
            metrics = val_stats(data['val']['y'], data['val']['c'], p)
            if best is None or rank(metrics) > rank(best['metrics']):
                best = dict(epoch=epoch+1, layer=layer, metrics=metrics)
                stale = 0
                atomic_torch(bestpath, dict(model=net.state_dict(), config_SHA256=contract_hash,
                                           feature_SHA256=pin['SHA256'], head=head, **best))
                np.savez(out / ('val_layer_' + str(layer) + '.npz'), probability=p,
                         position_id=np.asarray(data['val']['keys']))
            else: stale += 1
            atomic_torch(progress, dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                                       epoch=epoch+1, stale=stale, best=best, config_SHA256=contract_hash,
                                       feature_SHA256=pin['SHA256'], permutation_rng=generator.get_state(),
                                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                       python_rng=random.getstate(),numpy_rng=np.random.get_state(),batch_order=None,batch_offset=0))
            completed_epochs = epoch + 1
            resume_order,resume_offset=None,0
            save(out / 'PROGRESS.json', dict(layer=layer, epoch=epoch+1, best=best, stale=stale,
                                             job=os.environ['PBS_JOBID'], time=time.time()))
            if stale >= cfg['patience']: break
        save(report, dict(**best, trained_epochs=completed_epochs, config_SHA256=contract_hash,
                          feature_SHA256=pin['SHA256'], checkpoint_SHA256=sha(bestpath),
                          VAL_predictions_SHA256=sha(out / ('val_layer_' + str(layer) + '.npz'))))
        del net, optimizer, tx, vx; torch.cuda.empty_cache()
        # Private verified transport cache has no other consumers; source remains protected.
        local.unlink()
    candidates = [json.loads((out / ('layer_' + str(layer) + '.json')).read_text()) for layer in cfg['layers']]
    top = max(rank(x['metrics']) for x in candidates)
    winners = [x for x in candidates if rank(x['metrics']) == top]
    assert len(winners) == 1, 'Exact VAL AUROC/AP/BCE tie: retain candidates; do not choose arbitrary layer'
    winner = winners[0]; layer = winner['layer']
    p = np.load(out / ('val_layer_' + str(layer) + '.npz'), allow_pickle=False)['probability']
    sys.path.insert(0, str(REPO / 'code'))
    from baselines.streaming.stream_metrics import select_threshold
    threshold_y = select_threshold(data['val']['y'], p[:, 0])
    threshold_c = select_threshold(data['val']['c'], p[:, :2].sum(1))
    save(out / 'FROZEN_SELECTION.json', dict(arm=arm, selected=winner, threshold_step=threshold_y,
                                            threshold_prefix=threshold_c, config_SHA256=contract_hash,
                                            data_hashes=binding['data_hashes'], test_opened=False,
                                            source_manifest_SHA256=receipt_hash, status='FROZEN_WAIT_TEST'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
