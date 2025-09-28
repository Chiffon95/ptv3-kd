# src/train.py
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# AMP (PyTorch 2.8 권장 방식)
# SDPA 커널 제어: 반드시 모듈로 임포트해야 함
from torch.amp import GradScaler, autocast
try:
    import torch.backends.cuda.sdp_kernel as sdp_kernel
    HAS_SDPK = True
except ImportError:
    HAS_SDPK = False


from .dataset import S3DISRoomDataset, s3dis_collate_fn
from .model import PTv3KD
from .utils import (
    set_seed, AvgMeter, SegEvaluator, save_checkpoint, load_checkpoint,
    dump_json, log_scalar_dict
)

def subsample_batch(batch: Dict[str, torch.Tensor], max_points: int):
    if max_points <= 0:
        return batch
    N = batch["coord"].shape[0]
    if N <= max_points:
        return batch
    device = batch["coord"].device
    idx = torch.from_numpy(np.random.choice(N, size=max_points, replace=False)).to(device)
    out = {}
    for k in ["coord","feat","label","inst"]:
        out[k] = batch[k].index_select(0, idx)
    out["offset"] = torch.tensor([max_points], dtype=torch.int32, device=device)
    out["meta"] = batch["meta"]
    return out


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_loaders(cfg: Dict[str, Any]):
    dcfg = cfg["data"]; lcfg = cfg["loader"]
    root = dcfg["root"]
    train_areas = dcfg.get("train_areas", None)  # 예: ["Area_1"]
    val_areas   = dcfg.get("val_areas",   None)  # 예: ["Area_3"]

    train_ds = S3DISRoomDataset(root, split=dcfg["train_split"], strict=True, areas=train_areas)
    val_ds   = S3DISRoomDataset(root, split=dcfg["val_split"],   strict=True, areas=val_areas)

    train_loader = DataLoader(
        train_ds, batch_size=lcfg["batch_size"], shuffle=lcfg.get("shuffle_train", True),
        num_workers=lcfg["num_workers"], pin_memory=lcfg["pin_memory"],
        persistent_workers=lcfg.get("persistent_workers", True), collate_fn=s3dis_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=lcfg["batch_size"], shuffle=False,
        num_workers=lcfg["num_workers"], pin_memory=lcfg["pin_memory"],
        persistent_workers=lcfg.get("persistent_workers", True), collate_fn=s3dis_collate_fn
    )
    return train_loader, val_loader

def make_model(cfg: Dict[str, Any], num_classes: int):
    mcfg = cfg["model"]
    name = mcfg.get("name", "PTv3KD")
    if name != "PTv3KD":
        print(f"[warn] model.name={name} -> forcing PTv3KD")
    # teacher 1/3 스펙은 model.py의 _DEFAULT_CFG에 내장
    model = PTv3KD(num_classes=num_classes, cfg=None)
    return model

def make_optim(cfg: Dict[str, Any], model: nn.Module):
    ocfg = cfg["optim"]; scfg = cfg["sched"]
    if ocfg["name"].lower() == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(), lr=ocfg["lr"],
            betas=tuple(ocfg["betas"]), weight_decay=ocfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer {ocfg['name']}")
    # cosine per-epoch 스케줄러
    epochs = scfg["epochs"]; warm = scfg["warmup_epochs"]; min_lr = scfg["min_lr"]; base_lr = ocfg["lr"]
    def lr_lambda(epoch):
        if epoch < warm:
            return (epoch + 1) / max(1, warm)
        t = (epoch - warm) / max(1, epochs - warm)
        import math
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * t)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    return optim, sched

def make_loss(cfg: Dict[str, Any]):
    lcfg = cfg["loss"]
    if lcfg["name"] != "cross_entropy":
        print(f"[warn] loss.name={lcfg['name']} -> forcing cross_entropy")
    weight = lcfg.get("weight", None)
    weight_t = torch.tensor(weight, dtype=torch.float32) if isinstance(weight, list) else None
    ce = nn.CrossEntropyLoss(ignore_index=lcfg.get("ignore_index", -1), weight=weight_t)
    return ce

def to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, metrics_cfg: Dict[str, Any]):
    model.eval()
    evaluator = SegEvaluator(num_classes=num_classes, ignore_index=metrics_cfg.get("ignore_index", -1), device=device)
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        evaluator.add_batch(out["logits"], batch["label"])
    res = evaluator.compute(with_per_class=metrics_cfg.get("log", {}).get("per_class_iou", True))
    return res

def train_one_epoch(
    model, loader, device, optimizer, scaler, loss_fn,
    log_interval=50, grad_accum_steps=1, max_grad_norm=0.0,
    cfg_train_max_pts=0
):
    model.train()
    meter = AvgMeter()

    optimizer.zero_grad(set_to_none=True)
    step_in_accum = 0

    for it, batch in enumerate(loader, 1):
        batch = to_device(batch, device)

        # 🔧 서브샘플 적용 (YAML의 train.max_points 값 사용)
        batch = subsample_batch(batch, max_points=cfg_train_max_pts)
        
        # 🔧 디버그 로그: 큰 방을 만나면 N 출력 (최초 몇 번은 강제 찍기)
        if it <= 2 or (it % log_interval == 0):
            N = batch["coord"].shape[0]
            room = batch["meta"][0]["room"] if isinstance(batch["meta"], list) else ""
            print(f"[train] iter {it:05d} | N={N:,} | room={room}")

        with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            out = model(batch)
            loss = loss_fn(out["logits"], batch["label"])
            loss = loss / max(1, grad_accum_steps)

        scaler.scale(loss).backward()
        step_in_accum += 1

        if step_in_accum == grad_accum_steps:
            if max_grad_norm and max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

        meter.update(loss.item() * max(1, grad_accum_steps), n=1)

        if it % log_interval == 0:
            print(f"[train] iter {it:05d} | loss:{meter.avg:.4f}")
            meter.reset()

    if step_in_accum > 0:
        if max_grad_norm and max_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return meter.avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="path to configs/train_s3dis.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if (cfg["device"]["gpu"] and torch.cuda.is_available()) else "cpu")

    # ✅ SDPA 커널 설정: flash/mem-efficient 우선 사용 (해당 모듈이 있는 경우만)
    if HAS_SDPK:
        try:
            sdp_kernel.enable_flash_sdp(True)
            sdp_kernel.enable_mem_efficient_sdp(True)
            sdp_kernel.enable_math_sdp(False)
        except Exception as e:
            print(f"[warn] sdp_kernel setup skipped: {e}")
    else:
        print("[info] sdp_kernel module not available in this PyTorch build")

    # data / loaders
    train_loader, val_loader = make_loaders(cfg)
    num_classes = cfg["data"]["num_classes"]

    # model / optim / loss
    model = make_model(cfg, num_classes).to(device)
    optimizer, scheduler = make_optim(cfg, model)
    loss_fn = make_loss(cfg)
    if isinstance(loss_fn.weight, torch.Tensor):
        loss_fn.weight = loss_fn.weight.to(device)

    # amp scaler (torch.amp)
    scaler = GradScaler(device="cuda", enabled=cfg["train"]["amp"])

    # resume
    start_epoch = 0
    resume_path = cfg["train"].get("resume", "")
    if resume_path:
        print(f"[resume] loading from {resume_path}")
        try:
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler) + 1
            print(f"[resume] start from epoch {start_epoch}")
        except Exception as e:
            print(f"[resume] failed: {e}")

    epochs = cfg["sched"]["epochs"]
    best_metric_name = cfg["train"].get("save_best_metric", "sem_mIoU")
    best_metric = -1.0

    ckpt_dir = Path(cfg["ckpt"]["dir"])
    log_dir = Path(cfg["log"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    accum = int(cfg["train"].get("grad_accum_steps", 1))
    max_gn = float(cfg["train"].get("grad_clip_norm", 0.0))

    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} (lr={optimizer.param_groups[0]['lr']:.6f}) ===")

        avg_loss = train_one_epoch(
            model, train_loader, device, optimizer, scaler, loss_fn,
            log_interval=cfg["log"]["interval_iter"],
            grad_accum_steps=max(1, accum),
            max_grad_norm=max_gn,
            cfg_train_max_pts=int(cfg["train"].get("max_points", 0))
        )
        print(f"[train] epoch {epoch+1} | avg_loss:{avg_loss:.4f}")

        # scheduler step (per-epoch)
        scheduler.step()

        # validation
        if ((epoch + 1) % cfg["train"]["val_interval"]) == 0:
            val_res = validate(model, val_loader, device, num_classes, cfg["metrics"])
            print(f"[val] {log_scalar_dict(val_res)}")

            # save latest
            latest_path = ckpt_dir / cfg["ckpt"]["latest_name"]
            save_checkpoint(str(latest_path), model, optimizer, scheduler, epoch=epoch, extra={"val": val_res})

            # best by metric
            cur = float(val_res.get(best_metric_name, -1.0))
            if cur > best_metric:
                best_metric = cur
                best_path = ckpt_dir / cfg["ckpt"]["best_name"]
                save_checkpoint(str(best_path), model, optimizer, scheduler, epoch=epoch, extra={"val": val_res})
                dump_json(str(log_dir / "best_val.json"), {"epoch": epoch, "metric": best_metric_name, "value": best_metric})
                print(f"[ckpt] new best {best_metric_name}:{best_metric:.4f} -> saved {best_path.name}")

    print("\n[done] training finished.")

if __name__ == "__main__":
    main()
