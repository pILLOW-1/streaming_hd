"""Detector recipes; only explicit callers after dataset/runtime admission may fit.

The production runner never imports this module before all readiness checks pass.
Per-method objectives/optimizers remain separate. Tensor inputs contain no TEST
rows; callers provide only TRAIN and VAL to fit().
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from baselines.saplma.kernel import SAPLMA, KerasAdam
from baselines.streaming.kernel import LinearProbe, teacher_objective, prefix_objective
from baselines.icr_probe.kernel import ICRProbe


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class Progress:
    """Atomic epoch/update-boundary state including spent iterations and RNG."""
    def __init__(self, path, identity, deadline=float("inf"), worst_atomic_seconds=0.0):
        self.path, self.identity = Path(path), identity
        self.deadline, self.worst_atomic_seconds = deadline, worst_atomic_seconds
        self.stop_requested = False
        self.value = {"identity": identity, "stages": {}}
        if self.path.exists():
            self.value = torch.load(self.path, map_location="cpu", weights_only=False)
            if self.value["identity"] != identity:
                raise ValueError("CANDIDATE_RESUME_IDENTITY_MISMATCH")

    def restore(self, stage, model, optimizer=None, scheduler=None):
        value = self.value["stages"].get(stage)
        if value is None:
            return 0, None
        if value.get("atomic_in_progress"):
            raise ValueError("UNCOMMITTED_TRAINING_UNIT: preserve state; no silent epoch/update/line-search replay")
        model.load_state_dict(value["model"])
        if optimizer is not None and value.get("optimizer") is not None:
            optimizer.load_state_dict(value["optimizer"])
        if scheduler is not None and value.get("scheduler") is not None:
            scheduler.load_state_dict(value["scheduler"])
        restore_rng(value["rng"])
        return value["completed"], value

    def save(self, stage, completed, model, optimizer=None, scheduler=None, **extra):
        self.value["stages"][stage] = {"completed": completed, "model": copy.deepcopy(model.state_dict()),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": rng_state(), **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        torch.save(self.value, temporary)
        temporary.replace(self.path)
        from common.baseline_io import mirror_file
        mirror_file(self.path)

    def begin(self, stage, completed, model, optimizer=None, scheduler=None, **extra):
        if self.stop_requested or time.monotonic() + self.worst_atomic_seconds >= self.deadline:
            raise BoundaryExit("Verified checkpoint boundary; resume same candidate with full spent-unit state")
        self.save(stage, completed, model, optimizer, scheduler, atomic_in_progress=True, **extra)


class BoundaryExit(RuntimeError):
    """A legal walltime continuation boundary; never means candidate completion."""


def official_icr_val_bce(predictions, labels, batch_size=16):
    # Released ICR _validate_epoch averages per-batch means, including the
    # shorter last batch, rather than weighting those means by batch size.
    values = torch.as_tensor(predictions).float()
    targets = torch.as_tensor(labels).float()
    return float(torch.stack([F.binary_cross_entropy(values[i:i+batch_size], targets[i:i+batch_size])
                             for i in range(0, len(values), batch_size)]).mean())


def probabilities(model, x, device, sigmoid=True):
    model.eval()
    result = []
    with torch.no_grad():
        for begin in range(0, len(x), 2048):
            out = model(x[begin:begin + 2048].to(device)).flatten()
            result.append((out.sigmoid() if sigmoid else out).cpu())
    return torch.cat(result).numpy()


def linear_lbfgs(x, labels, width, device, progress, stage, strong_wolfe):
    model = LinearProbe(width).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100,
                                 line_search_fn="strong_wolfe" if strong_wolfe else None)
    completed, _ = progress.restore(stage, model, optimizer)
    if completed:
        return model
    # LBFGS's internal line-search locals cannot be checkpointed mid-step. An
    # interruption of this explicitly marked whole step must fail closed.
    progress.begin(stage, 0, model, optimizer)
    features, targets = x.to(device), labels.to(device)
    def closure():
        optimizer.zero_grad()
        loss = teacher_objective(model, features, targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    progress.save(stage, 1, model, optimizer, atomic_in_progress=False)
    return model


def fit_vector(config, train, val, labels, val_labels, device, progress):
    """Exact SAPLMA / ICR or linear source recipes on frozen feature vectors."""
    method = config["method_key"]
    x, xv = torch.as_tensor(train["x"]).float(), torch.as_tensor(val["x"]).float()
    y, yv = torch.as_tensor(labels).float(), torch.as_tensor(val_labels).float()
    if method == "global_mean" or (method == "streaming" and config["level"] != "prefix"):
        if config.get("select_weights_by_val_auroc", False):
            from baselines.streaming.observed_lbfgs import ObservedLBFGS
            from baselines.streaming.run import evaluate_metrics
            from common.checkpoint_selection import best_snapshot
            model=LinearProbe(x.shape[1]).to(device)
            normalized=F.normalize(x,dim=1).to(device)
            normalized_val=F.normalize(xv,dim=1)
            best=None
            def accepted(iteration):
                nonlocal best
                prediction=probabilities(model,normalized_val,device)
                metric=evaluate_metrics(np.asarray(val_labels),prediction,.5)
                best=best_snapshot((metric['AUROC'],metric['AP']),model.state_dict(),best)
            optimizer=ObservedLBFGS(model.parameters(),on_accepted=accepted,lr=1.,max_iter=100,
                line_search_fn="strong_wolfe" if config["level"]!="prefix" else None)
            completed, _ = progress.restore("linear_selection",model,optimizer)
            if completed:
                return model,{"normalize":True,"sigmoid":True}
            progress.begin("linear_selection",0,model,optimizer)
            def closure():
                optimizer.zero_grad();loss=teacher_objective(model,normalized,y.to(device));loss.backward();return loss
            optimizer.step(closure)
            if best is None or best['tie']:raise ValueError("No unique best VAL AUROC/AP linear update")
            model.load_state_dict(best['state'])
            progress.save("linear_selection",1,model,optimizer,atomic_in_progress=False)
            return model,{"normalize":True,"sigmoid":True}
        return linear_lbfgs(F.normalize(x, dim=1), y, x.shape[1], device, progress,
                            "linear", config["level"] != "prefix"), {"normalize": True, "sigmoid": True}
    if method == "saplma":
        model = SAPLMA(x.shape[1]).to(device)
        optimizer = KerasAdam(model.parameters())
        scheduler = None
        epochs, batch, sigmoid = 5, 32, True
    elif method == "icr_probe":
        model = ICRProbe(x.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=5)
        epochs, batch, sigmoid = 100, 16, False
    else:
        raise ValueError(f"Wrong vector recipe: {method}")
    completed, old = progress.restore("fit", model, optimizer, scheduler)
    best = old.get("best") if old else None
    best_loss = old.get("best_loss", float("inf")) if old else float("inf")
    best_rank = old.get("best_rank", (-float("inf"), -float("inf"))) if old else (-float("inf"), -float("inf"))
    best_tie = old.get("best_tie", False) if old else False
    for epoch in range(completed, epochs):
        progress.begin("fit", epoch, model, optimizer, scheduler, best=best, best_loss=best_loss)
        model.train()
        order = torch.randperm(len(x))
        for begin in range(0, len(order), batch):
            ids = order[begin:begin + batch]
            if method == "icr_probe" and len(ids) == 1:
                raise ValueError("OFFICIAL_ICR_BATCHNORM_SINGLETON: explicit batch policy required, no silent drop")
            optimizer.zero_grad()
            out = model(x[ids].to(device)).flatten()
            target = y[ids].to(device)
            loss = F.binary_cross_entropy_with_logits(out, target) if sigmoid else F.binary_cross_entropy(out, target)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            pred = probabilities(model, xv, device, sigmoid=False)
            value = official_icr_val_bce(pred, yv, batch)
            if value < best_loss:
                best_loss = value
                if not config.get("select_weights_by_val_auroc", False):
                    best = copy.deepcopy(model.state_dict())
            scheduler.step(value)
        if config.get("select_weights_by_val_auroc", False):
            from baselines.streaming.run import evaluate_metrics
            pred = probabilities(model, xv, device, sigmoid=sigmoid)
            metric = evaluate_metrics(np.asarray(val_labels), pred, .5)
            rank = (metric["AUROC"], metric["AP"])
            if rank > best_rank:
                best_rank, best, best_tie = rank, copy.deepcopy(model.state_dict()), False
            elif rank == best_rank and any(not torch.equal(best[k], v) for k, v in model.state_dict().items()):
                best_tie = True
        progress.save("fit", epoch + 1, model, optimizer, scheduler, best=best, best_loss=best_loss, best_rank=best_rank, best_tie=best_tie)
    if config.get("select_weights_by_val_auroc", False) and best_tie:
        raise ValueError("Exact best VAL AUROC/AP checkpoint tie; no earlier-update preference permitted")
    if best is not None:
        model.load_state_dict(best)
    return model, {"normalize": False, "sigmoid": sigmoid}


def contiguous_trajectories(meta):
    groups = []
    for index, row in enumerate(meta):
        if not groups or meta[groups[-1][0]]["tid"] != row["tid"]:
            groups.append([])
        groups[-1].append(index)
    if len({meta[g[0]]["tid"] for g in groups}) != len(groups):
        raise ValueError("Streaming requires complete contiguous trajectories, no repeated fragments")
    return groups


def fit_prefix(train, labels, train_meta, teacher_labels, device, progress):
    x = F.normalize(torch.as_tensor(train["x"]).float(), dim=1)
    xs = F.normalize(torch.as_tensor(train["teacher_x"]).float(), dim=1)
    y, ys = torch.as_tensor(labels).float(), torch.as_tensor(teacher_labels).float()
    teacher = linear_lbfgs(xs, ys, xs.shape[1], device, progress, "teacher", True)
    baseline = linear_lbfgs(x, y, x.shape[1], device, progress, "baseline", False)
    model = copy.deepcopy(baseline)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # official default weight_decay=.01
    groups = contiguous_trajectories(train_meta)
    completed, _ = progress.restore("prefix", model, optimizer)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for epoch in range(completed, 20):
        progress.begin("prefix", epoch, model, optimizer)
        model.train()
        order = list(range(len(groups)))
        random.shuffle(order)
        for begin in range(0, len(order), 64):
            selected = [groups[j] for j in order[begin:begin + 64]]
            indices = [i for group in selected for i in group]
            optimizer.zero_grad()
            with torch.no_grad():
                alarm = teacher(xs[indices].to(device)).sigmoid()
            loss = prefix_objective(model(x[indices].to(device)), y[indices].to(device), alarm, list(map(len, selected)))
            loss.backward()
            optimizer.step()
        progress.save("prefix", epoch + 1, model, optimizer)
    return model, {"normalize": True, "sigmoid": True}


class TokenBags:
    """Read exactly one native token bag at a time; no full-bank RAM copy."""
    def __init__(self,pack):
        self.tokens=pack['tokens'];self.offsets=np.asarray(pack['offsets'],dtype=np.int64)
        if self.offsets[0]!=0 or self.offsets[-1]!=len(self.tokens) or np.any(np.diff(self.offsets)<=0):
            raise ValueError('Exact nonempty ragged token spans required')
        self.mode=pack.get('hami_mode','se');self.dtype=pack.get('token_dtype','float')
        if self.mode not in ('ori','se'):raise ValueError('Unsupported HaMI branch')
        self.probability=None
        if self.mode=='se':
            self.probability=np.asarray(pack['semantic_probability'])
            if len(self.probability)!=len(self) or not np.all((0<=self.probability)&(self.probability<=1)):
                raise ValueError('Aligned real semantic probabilities required for SE')
        if self.dtype=='bfloat16_bits' and np.dtype(self.tokens.dtype)!=np.dtype('uint16'):
            raise ValueError('BF16 bit bank must use uint16 storage')
    def __len__(self):return len(self.offsets)-1
    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        a,b=self.offsets[index:index+2]
        value=torch.from_numpy(np.array(self.tokens[a:b],copy=True))
        if isinstance(self.tokens,np.memmap):
            import mmap
            self.tokens._mmap.madvise(mmap.MADV_DONTNEED)
        if self.dtype=='bfloat16_bits':value=value.view(torch.bfloat16)
        value=value.float()[:,None,:]
        return value if self.mode=='ori' else value*(1+float(self.probability[index]))

def token_bags(pack):
    return TokenBags(pack)


def hami_scores(model, pack, device):
    bags = token_bags(pack)
    model.eval()
    result = []
    with torch.no_grad():
        for bag in bags:
            scores, _, _ = model([bag], [], n_layer=0, device=device)
            k = int(len(scores) * .1) + 1
            result.append(float(torch.topk(scores, k).values.mean()))
    return np.asarray(result)


def fit_hami(train, val, labels, val_labels, device, progress):
    from baselines.hami.kernel import HaMI, MIL_loss
    from sklearn.metrics import roc_auc_score
    bags = token_bags(train)
    normal = [i for i, y in enumerate(labels) if y == 0]
    abnormal = [i for i, y in enumerate(labels) if y == 1]
    if min(len(normal), len(abnormal)) < 128:
        raise ValueError("HaMI original balanced batch128+128 requires enough TRAIN bags; no hidden recipe change")
    model = HaMI(bags[0].shape[-1], 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.1, patience=20, min_lr=1e-6)
    done, old = progress.restore("hami", model, optimizer, scheduler)
    best_auc = old.get("best_auc", -float("inf")) if old else -float("inf")
    best_ap = old.get("best_ap", -float("inf")) if old else -float("inf")
    best_tie = old.get("best_tie", False) if old else False
    best = old.get("best") if old else None
    # Preserve source DataLoader cycles (shuffle, drop_last) across updates.
    cycles = old.get("cycles") if old else {"normal": [], "abnormal": []}
    for update in range(done, 100):
        progress.begin("hami", update, model, optimizer, scheduler, best_auc=best_auc, best=best, cycles=cycles)
        selected = []
        for name, population in (("normal", normal), ("abnormal", abnormal)):
            if len(cycles[name]) < 128:
                cycles[name] = list(population)
                random.shuffle(cycles[name])
                cycles[name] = cycles[name][:len(cycles[name]) // 128 * 128]
            selected.append([bags[i] for i in cycles[name][:128]])
            cycles[name] = cycles[name][128:]
        model.train()
        optimizer.zero_grad()
        scores, nl, al = model(*selected, n_layer=0, device=device)
        loss = MIL_loss(scores, nl, al, device, .1)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        from baselines.streaming.run import evaluate_metrics
        metric=evaluate_metrics(np.asarray(val_labels),hami_scores(model,val,device),.5)
        auc,ap=metric['AUROC'],metric['AP']
        if (auc,ap)>(best_auc,best_ap):
            best_auc,best_ap,best,best_tie=auc,ap,copy.deepcopy(model.state_dict()),False
        elif (auc,ap)==(best_auc,best_ap) and any(not torch.equal(best[k],v) for k,v in model.state_dict().items()):
            best_tie=True
        progress.save("hami", update + 1, model, optimizer, scheduler,
                      best_auc=best_auc,best_ap=best_ap,best_tie=best_tie, best=best, cycles=cycles)
    if best_tie:raise ValueError("Exact best VAL AUROC/AP HaMI checkpoint tie")
    model.load_state_dict(best)
    return model, {"family": "hami"}


def fit_ars(train, val, labels, val_labels, device, progress, select_weights_by_val_auroc=False):
    """Official answer-agreement shaping then official nonlinear supervised probe."""
    from baselines.ars.kernel import LinearMap, icm_sameid_ars_loss
    from baselines.ars.detector_kernel import NonLinearClassifier
    from sklearn.metrics import log_loss
    projection = LinearMap(train["variants_x"].shape[1], 128).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=.001, weight_decay=1e-4)
    x = torch.as_tensor(train["variants_x"]).float()
    ids = torch.as_tensor(train["variant_original_id"]).long()
    agreement = torch.as_tensor(train["variant_agreement"]).bool()
    batches = len(x) // 32
    if batches < 1:
        raise ValueError("ARS requires original frozen TRAIN answer-agreement variants")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batches * 50, eta_min=1e-6)
    completed, _ = progress.restore("ars_projection", projection, optimizer, scheduler)
    for epoch in range(completed, 50):
        progress.begin("ars_projection", epoch, projection, optimizer, scheduler)
        order = torch.randperm(len(x))
        projection.train()
        for begin in range(0, batches * 32, 32):
            selected = order[begin:begin + 32]
            optimizer.zero_grad()
            loss = icm_sameid_ars_loss(projection(x[selected].to(device)), ids[selected].to(device),
                                     agreement[selected].to(device), temperature=.1, lambda_reg=.001, W=projection.W)
            loss.backward()
            optimizer.step()
            scheduler.step()
        progress.save("ars_projection", epoch + 1, projection, optimizer, scheduler)
    projection.eval()
    with torch.no_grad():
        tx = projection(torch.as_tensor(train["x"]).float().to(device)).cpu()
        vx = projection(torch.as_tensor(val["x"]).float().to(device)).cpu()
    head = NonLinearClassifier(128).to(device)  # official p_drop=.4, BN momentum=.05, noise .008
    optimizer = torch.optim.SGD(head.parameters(), lr=.05, momentum=.9, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=7, min_lr=1e-4)
    completed, old = progress.restore("ars_detector", head, optimizer, scheduler)
    best_loss = old.get("best_loss", float("inf")) if old else float("inf")
    best = old.get("best") if old else None
    y = torch.as_tensor(labels).float()
    best_rank = old.get("best_rank", (-float("inf"), -float("inf"))) if old else (-float("inf"), -float("inf"))
    best_tie = old.get("best_tie", False) if old else False
    for epoch in range(completed, 100):
        progress.begin("ars_detector", epoch, head, optimizer, scheduler, best_loss=best_loss, best=best)
        head.train()
        order = torch.randperm(len(tx))
        for begin in range(0, len(tx), 128):
            index = order[begin:begin + 128]
            if len(index) == 1:
                raise ValueError("ARS BatchNorm singleton batch requires explicit admission policy")
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(head(tx[index].to(device)).flatten(), y[index].to(device))
            loss.backward()
            optimizer.step()
        prediction = probabilities(head, vx, device)
        value = float(log_loss(val_labels, np.asarray(prediction,dtype=np.float64)))
        if value < best_loss:
            best_loss = value
            if not select_weights_by_val_auroc:best = copy.deepcopy(head.state_dict())
        if select_weights_by_val_auroc:
            from baselines.streaming.run import evaluate_metrics
            metric=evaluate_metrics(np.asarray(val_labels),prediction,.5)
            rank=(metric['AUROC'],metric['AP'])
            if rank>best_rank:best_rank,best,best_tie=rank,copy.deepcopy(head.state_dict()),False
            elif rank==best_rank and any(not torch.equal(best[k],v) for k,v in head.state_dict().items()):best_tie=True
        scheduler.step(value)
        progress.save("ars_detector", epoch + 1, head, optimizer, scheduler, best_loss=best_loss, best=best, best_rank=best_rank, best_tie=best_tie)
    # Official train_probing.py saves the final epoch, not the best VAL epoch.
    # VAL loss controls only the released scheduler; layer selection is separate.
    if select_weights_by_val_auroc:
        if best_tie:raise ValueError("Exact best VAL AUROC/AP checkpoint tie")
        head.load_state_dict(best)
    return {"projection": projection, "head": head}, {"family": "ars"}
