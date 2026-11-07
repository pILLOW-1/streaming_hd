#!/usr/bin/env python3
"""Unchanged predecessor Streaming metrics and threshold algorithm."""
import os
from pathlib import Path
import numpy as np

def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores > threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    positive = int(y.sum())
    negative = len(y)-positive
    order = np.argsort(-scores, kind='stable')
    sy, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0)+1, len(y)]
    ctp = np.cumsum(sy)[ends-1]
    cfp = ends-ctp
    recall = ctp/positive if positive else np.zeros_like(ctp, dtype=float)
    precision = ctp/ends
    auc = float(np.trapezoid(np.r_[0,recall], np.r_[0,cfp/negative])) if positive and negative else None
    ap = float(np.sum(np.diff(np.r_[0,recall])*precision)) if positive else None
    return {'AUROC':auc,'AP':ap,'ACC':(tp+tn)/len(y),
        'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/positive if positive else None,
        'FPR':fp/negative if negative else None,'TP':tp,'FP':fp,'TN':tn,'FN':fn,
        'steps':len(y),'positive':positive,'negative':negative,'missing':0,'threshold':float(threshold)}

def select_threshold(y, scores):
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores,dtype=np.float64)
    assert y.sum() > 0 and np.isfinite(scores).all()
    order = np.argsort(-scores,kind='stable')
    y, scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0)+1,len(y)]
    ctp = np.cumsum(y)[ends-1]
    candidates = [(float(scores[0]), 0, 0)]
    for i, end in enumerate(ends):
        threshold = float(scores[end]) if end < len(y) else float(np.nextafter(scores[-1],-np.inf))
        candidates.append((threshold,int(ctp[i]),int(end)))
    positive = int(y.sum())
    ranked = []
    for threshold,tp,count in candidates:
        f1 = 2*tp/(positive+count) if positive+count else 0.
        precision = tp/count if count else 0.
        ranked.append(((f1,precision,threshold),threshold))
    return max(ranked)[1]

def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez(stream,**arrays)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
