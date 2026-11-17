"""Train on train; select on validation; never evaluate test in this module."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .common import (PROTOCOL,Paused,atomic_json,atomic_torch,artifact,verify_artifacts,
                     cpu,lock,read_json,restore_rng,rng_state,runtime,seed_all)
from .config import selection_value
from .metrics import evaluate_model
from .model import build_model

def objective(sl,pl,ys,yp,prefix_weight):
    if not np.isfinite(prefix_weight) or prefix_weight<0: raise ValueError("Invalid prefix loss weight")
    return nn.functional.binary_cross_entropy_with_logits(sl,ys) + \
           prefix_weight*nn.functional.binary_cross_entropy_with_logits(pl,yp)

def train_run(args,dataset,config,package_sha,control):
    directory=Path(args.run_dir).resolve(); directory.mkdir(parents=True,exist_ok=True)
    identity=dict(source=dataset.identity,package_sha256=package_sha,config=config,
                  dataset=args.dataset,seed=args.seed,smoke=bool(args.smoke))
    env=runtime(args.device,args.threads)
    with lock(directory/".train.lock"):
        completed=directory/"result.json"
        if completed.exists():
            result=read_json(completed)
            if result["identity"]!=identity or result["status"]!="VALIDATION_COMPLETE":
                raise ValueError("Existing result belongs to another experiment")
            verify_artifacts(directory,result["artifacts"])
            atomic_json(directory/"status.json",dict(status="completed",dataset=args.dataset,
                        variant=args.variant,seed=args.seed,verified_reused_result=True))
            print("Verified completed validation run:",directory,flush=True)
            return result
        seed_all(args.seed)
        model=build_model(dataset.width,config).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
        train=dataset.split("train",args.smoke); val=dataset.split("val",args.smoke)
        if not train or not val: raise ValueError("Empty train/validation split")
        progress=dict(stage="train",epoch=1,cursor=0,updates=0,order=list(range(len(train))),
                      order_epoch=0,loss_sum=0.0,stale=0,best_value=-1.0,best_epoch=None,
                      best_state=None,history=[],elapsed_training_seconds=0.0)
        last=directory/"last.pt"
        if last.exists():
            saved=torch.load(last,map_location="cpu",weights_only=True)
            if saved.get("identity")!=identity or saved.get("runtime")!=env:
                raise ValueError("Resume requires the same code, source, configuration, seed and runtime versions/device type")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            progress=saved["progress"]; restore_rng(saved["rng"])
        elif any(p.name not in (".train.lock","status.json","last.pt.partial") for p in directory.iterdir()):
            raise ValueError("Existing files without a resumable checkpoint; use a new output directory")
        started=time.monotonic(); previous_elapsed=progress["elapsed_training_seconds"]
        def publish(status,**extra):
            atomic_json(directory/"status.json",dict(status=status,dataset=args.dataset,variant=args.variant,
                        seed=args.seed,smoke_run=bool(args.smoke),epoch=progress["epoch"],stage=progress["stage"],
                        next_trajectory=progress["cursor"],updates=progress["updates"],**extra))
        def save():
            progress["elapsed_training_seconds"]=previous_elapsed+time.monotonic()-started
            atomic_torch(last,dict(format_version=PROTOCOL,identity=identity,runtime=env,
                         model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),
                         progress=cpu(progress),rng=rng_state()))
            publish("running")
        save()
        try:
            while progress["stage"]!="done":
                control.check()
                if progress["stage"]=="train":
                    if progress["order_epoch"]!=progress["epoch"]:
                        np.random.shuffle(progress["order"]); progress["order_epoch"]=progress["epoch"]
                    model.train()
                    while progress["cursor"]<len(train):
                        control.check()
                        row=train[progress["order"][progress["cursor"]]]
                        tr=dataset.load(row)
                        optimizer.zero_grad(set_to_none=True)
                        sl,pl=model.forward_trajectory(tr,args.device)
                        loss=objective(sl,pl,tr.step_labels.to(args.device),tr.prefix_labels.to(args.device),
                                       config["prefix_loss_weight"])
                        if not bool(torch.isfinite(loss)): raise ValueError("Nonfinite loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),config["clip_norm"],error_if_nonfinite=True)
                        optimizer.step()
                        progress["loss_sum"]+=float(loss.detach().cpu())
                        progress["cursor"]+=1; progress["updates"]+=1; control.updates+=1
                        del sl,pl,loss,tr
                        if progress["updates"]%50==0: save()
                        control.check()
                    progress["stage"]="validate"; save()
                if progress["stage"]=="validate":
                    scores,_=evaluate_model(model,dataset,val,args.device,control)
                    # A tiny smoke may have one class; it tests execution, never selection quality.
                    undefined=any(scores[t]["AUC"] is None for t in ("step","prefix"))
                    if undefined and args.smoke: value=0.0
                    else: value=selection_value(scores,config["selection"])
                    progress["history"].append(dict(epoch=progress["epoch"],loss=progress["loss_sum"]/len(train),
                        validation=scores,selection_value=value,smoke_selection_undefined=undefined))
                    if value>progress["best_value"]+config["minimum_improvement"]:
                        progress.update(best_value=value,best_epoch=progress["epoch"],best_state=cpu(model.state_dict()),stale=0)
                    else: progress["stale"]+=1
                    print(f"{args.variant} seed={args.seed} epoch={progress['epoch']} "
                          f"val step={scores['step']['AUC']} prefix={scores['prefix']['AUC']} selection={value:.6f}",flush=True)
                    done=progress["epoch"]>=config["epochs"] or progress["stale"]>=config["patience"]
                    progress.update(stage="done" if done else "train",epoch=progress["epoch"]+1,cursor=0,loss_sum=0.0)
                    save()
            control.check()
            model.load_state_dict(progress["best_state"])
            scores,records=evaluate_model(model,dataset,val,args.device,control)
            selection=dict(metric=config["selection"],epoch=progress["best_epoch"],value=progress["best_value"])
            atomic_json(directory/"validation_predictions.json",records)
            atomic_torch(directory/"checkpoint.pt",dict(format_version=PROTOCOL,identity=identity,
                         selection=selection,architecture=model.architecture_config,state_dict=progress["best_state"],
                         hidden_dim=dataset.width))
            result=dict(status="VALIDATION_COMPLETE",format_version=PROTOCOL,dataset=args.dataset,
                        variant=args.variant,seed=args.seed,smoke_run=bool(args.smoke),
                        benchmark_eligible=not args.smoke and dataset.identity.get("kind")!="synthetic_test_fixture",
                        evaluation_split="val",identity=identity,runtime=env,
                        selection=selection,validation=scores,history=progress["history"],
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        elapsed_training_seconds=progress["elapsed_training_seconds"],
                        train_trajectories=len(train),validation_trajectories=len(val),
                        test_evaluated=False,answer_supervision_used_for_training=False,
                        prefix_transfer_is_unchanged_prefix_score=True,
                        graph_edges_are_causal_attributions=False,
                        artifacts={"checkpoint":artifact(directory,"checkpoint.pt"),
                                   "validation_predictions":artifact(directory,"validation_predictions.json")})
            atomic_json(completed,result); publish("completed",selection=selection)
            return result
        except Paused as exc:
            save(); publish("paused",reason=str(exc),resumable_checkpoint=True)
            raise
        except Exception as exc:
            publish("failed",error=f"{type(exc).__name__}: {exc}",resumable_checkpoint=last.exists())
            raise
