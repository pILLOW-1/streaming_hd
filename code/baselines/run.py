#!/usr/bin/env python3
"""Run baseline detectors against the agreed local dataset and native feature packs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))


def target_rows(rows, level):
    keys, labels = [], []
    for row in rows:
        if level == 'answer':
            keys.append(row['sample_id'] + ':answer')
            labels.append(row['answer_hallucination'])
        else:
            for original, value in zip(row['retained_original_step_ids'], row[level+'_hallucination']):
                keys.append(f'{row["sample_id"]}:{original}')
                labels.append(value)
    return keys, labels


def load_pack(folder, rows, level, method=None, layer=None):
    """Bind every native feature to its original trajectory/Step, never row count alone."""
    import numpy as np
    path = folder.with_suffix('.npz')
    manifest = json.loads(folder.with_suffix('.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest['SHA256'] != digest:
        raise ValueError(f'Feature pack hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as source:
        pack = {key: source[key] for key in source.files}
    keys, y = target_rows(rows, level)
    if list(map(str, pack.pop('position_id'))) != keys:
        raise ValueError('Feature pack does not cover the exact ordered dataset positions')
    if method is not None and manifest.get('method') != method:
        raise ValueError('Feature method identity mismatch')
    if layer is not None and manifest.get('layer') != layer:
        raise ValueError('Feature layer identity mismatch')
    for name in ('x', 'score', 'components', 'semantic_probability'):
        if name in pack and len(pack[name]) != len(keys):
            raise ValueError('Feature row count mismatch: ' + name)
    if manifest['level'] != level:
        raise ValueError('Feature target level mismatch')
    if any(k in pack for k in ('labels', 'gold_answer', 'step_hallucination', 'prefix_hallucination', 'answer_hallucination')):
        raise ValueError('Feature packs must not contain supervised targets or reference answers')
    for name, value in pack.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite feature: {name}')
    if method=='hami':
        pack['hami_mode']=manifest.get('hami_mode','se')
        pack['token_dtype']=manifest.get('token_dtype','float')
    return pack, np.asarray(y, dtype=np.int64), keys, digest


def raw_scores(method, pack, cfg):
    import numpy as np
    if method == 'llm_check':
        return np.asarray(pack['score']).reshape(-1) * cfg['score_sign']
    if method == 'race':
        from baselines.race.adapter import score_components
        names = ('consistency','cross_uncertainty','sindex','entity_consistency')
        return np.asarray([score_components(dict(zip(names, row))) for row in pack['components']]) * cfg['score_sign']
    if method == 'rhd':
        from baselines.rhd.kernel import score
        return score(pack['components'], cfg['weights']) * cfg['score_sign']
    raise ValueError(f'Unknown raw score baseline: {method}')


def predict(model, pack, options, device):
    import torch
    from common.baseline_training import probabilities, hami_scores
    if options.get('family') == 'hami':
        return hami_scores(model, pack, device)
    x = torch.as_tensor(pack['x']).float()
    if options.get('family') == 'ars':
        model['projection'].eval()
        with torch.no_grad():
            x = model['projection'](x.to(device)).cpu()
        return probabilities(model['head'], x, device)
    if options.get('normalize'):
        x = torch.nn.functional.normalize(x, dim=1)
    return probabilities(model, x, device, options.get('sigmoid', True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    import numpy as np
    import torch
    from common.dataset import load_cell
    from common.baseline_training import Progress, seed_all, fit_vector, fit_hami, fit_ars
    from baselines.streaming.run import evaluate_metrics, choose, save
    from baselines.streaming.stream_metrics import select_threshold

    paths, cfg = json.loads(args.paths.read_text()), json.loads(args.recipe.read_text())
    method, level = cfg['method_key'], cfg['level']
    if method in ('streaming','rede'):
        parser.error('Use the dedicated method entry point for this multi-stage method.')
    if not paths.get('baseline_feature_root') or not paths.get('output_root'):
        parser.error('Set baseline_feature_root and output_root in the private paths file.')
    rows = load_cell(paths, args.model, args.domain)
    out = Path(paths['output_root'])/method/args.model/args.domain/level
    out.mkdir(parents=True, exist_ok=False)
    save(out/'recipe.json', cfg)
    feature_root = Path(paths['baseline_feature_root'])/method/args.model/args.domain/level
    if cfg['layers'] == 'all':
        experiment=json.loads((ROOT/'experiments/streaming_622_p70/config.json').read_text())
        cfg['layers']=next(c['candidate_layers'] for c in experiment['cells'] if (c['model'],c['domain'],c['level'])==(args.model,args.domain,'step'))
        if method=='llm_check' and args.model.startswith('qwen'):
            cfg['layers']=[i for i in cfg['layers'] if i%4==0]
    if not cfg['layers']:raise ValueError('At least one declared candidate layer required')
    save(out/'effective_recipe.json', cfg)
    candidates = []
    bundles = {}
    for layer in cfg['layers']:
        seed_all(cfg.get('seed',42))
        directory = feature_root/f'layer_{layer}'
        train, y, _, train_hash = load_pack(directory/'train',rows['train'],level,method,layer)
        val, vy, _, val_hash = load_pack(directory/'val',rows['val'],level,method,layer)
        if len(set(y.tolist())) != 2 or len(set(vy.tolist())) != 2:
            raise ValueError('Both classes required in TRAIN and VAL')
        model, options = None, {}
        if method in ('llm_check','race','rhd'):
            scores = raw_scores(method,val,cfg)
        else:
            identity = dict(method=method,model=args.model,domain=args.domain,level=level,
                            layer=layer,train_SHA256=train_hash,val_SHA256=val_hash,recipe=cfg)
            progress = Progress(out/f'progress_layer_{layer}.pt',identity)
            if method == 'hami':
                if train['hami_mode']!=cfg.get('hami_mode','se') or val['hami_mode']!=cfg.get('hami_mode','se'):
                    raise ValueError('HaMI feature branch differs from declared ori/SE recipe')
                model,options = fit_hami(train,val,y,vy,args.device,progress)
            elif method == 'ars':
                model,options = fit_ars(train,val,y,vy,args.device,progress,select_weights_by_val_auroc=cfg.get("select_weights_by_val_auroc",False))
            else:
                model,options = fit_vector(cfg,train,val,y,vy,args.device,progress)
            scores = predict(model,val,options,args.device)
        if scores.shape != vy.shape or not np.isfinite(scores).all():
            raise ValueError('Invalid VAL predictions')
        metrics = evaluate_metrics(vy,scores,.5)
        candidates.append(dict(layer=layer,AUROC=metrics['AUROC'],AP=metrics['AP']))
        # Only detector state and compact VAL scores are retained between layers.
        if model is not None:
            state = ({k:v.state_dict() for k,v in model.items()} if isinstance(model,dict) else model.state_dict())
            torch.save(state,out/f'layer_{layer}.pt')
            for module in (model.values() if isinstance(model,dict) else [model]):
                module.to('cpu')
        bundles[layer] = (model,options,scores)
    selected = choose(candidates)
    layer = selected['layer'];model,options,vs = bundles[layer]
    if model is not None:
        for module in (model.values() if isinstance(model,dict) else [model]):
            module.to(args.device)
    _,vy,vkeys,vhash = load_pack(feature_root/f'layer_{layer}'/'val',rows['val'],level,method,layer)
    threshold = select_threshold(vy,vs)
    save(out/'FROZEN_SELECTION.json',dict(selected=selected,threshold=threshold,VAL=evaluate_metrics(vy,vs,threshold)))
    # TEST is opened only after selection and operating point have been frozen.
    test,ty,tkeys,thash = load_pack(feature_root/f'layer_{layer}'/'test',rows['test'],level,method,layer)
    if method=='hami' and test['hami_mode']!=cfg.get('hami_mode','se'):
        raise ValueError('HaMI TEST branch differs from frozen recipe')
    ts = raw_scores(method,test,cfg) if model is None else predict(model,test,options,args.device)
    if ts.shape != ty.shape or not np.isfinite(ts).all():
        raise ValueError('Invalid TEST predictions')
    np.savez(out/'test_predictions.npz',position_id=np.asarray(tkeys),labels=ty,scores=ts)
    save(out/'result.json',dict(method=method,model=args.model,domain=args.domain,level=level,
         selected=selected,threshold=threshold,TEST=evaluate_metrics(ty,ts,threshold),test_features_SHA256=thash))
    save(out/'candidates.json',candidates)
    print(json.dumps(dict(status='COMPLETE',output=str(out))))


if __name__ == '__main__':
    main()
