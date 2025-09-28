import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------
# Simple meters / timer
# -------------------------
class AvgMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)


class Timer:
    def __init__(self):
        self.t0 = time.time()
    def reset(self):
        self.t0 = time.time()
    def elapsed(self):
        return time.time() - self.t0


# -------------------------
# Checkpoint I/O
# -------------------------
def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    epoch: int = 0,
                    extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))


# -------------------------
# Metrics (Semantic Seg)
# -------------------------
@dataclass
class Confusion:
    mat: torch.Tensor  # (C, C) long: row=gt, col=pred
    ignore_index: int = -1

    @staticmethod
    def empty(num_classes: int, ignore_index: int = -1, device: torch.device = torch.device("cpu")):
        return Confusion(torch.zeros((num_classes, num_classes), dtype=torch.long, device=device),
                         ignore_index=ignore_index)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: (N,) int64
        target: (N,) int64
        """
        assert pred.shape == target.shape
        C = self.mat.size(0)

        mask = target != self.ignore_index
        t = target[mask].clamp(0, C - 1).to(torch.long)
        p = pred[mask].clamp(0, C - 1).to(torch.long)
        idx = t * C + p
        binc = torch.bincount(idx, minlength=C * C)
        self.mat += binc.view(C, C)

    def merge(self, other: "Confusion"):
        self.mat += other.mat

    # ---- Derived metrics ----
    def per_class_iou(self) -> torch.Tensor:
        C = self.mat.size(0)
        tp = self.mat.diag().to(torch.float64)
        fp = self.mat.sum(dim=0).to(torch.float64) - tp
        fn = self.mat.sum(dim=1).to(torch.float64) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom.clamp_min(1e-12), torch.zeros_like(tp))
        return iou  # (C,)

    def per_class_acc(self) -> torch.Tensor:
        tp = self.mat.diag().to(torch.float64)
        total = self.mat.sum(dim=1).to(torch.float64)
        acc = torch.where(total > 0, tp / total.clamp_min(1e-12), torch.zeros_like(tp))
        return acc  # (C,)

    def point_acc(self) -> float:
        return (self.mat.diag().sum().to(torch.float64) / self.mat.sum().clamp_min(1e-12)).item()

    def sem_mIoU(self) -> float:
        return self.per_class_iou().mean().item()

    def sem_mAcc(self) -> float:
        return self.per_class_acc().mean().item()

    def fwIoU(self) -> float:
        iou = self.per_class_iou()
        freq = self.mat.sum(dim=1).to(torch.float64)
        w = freq / freq.sum().clamp_min(1e-12)
        return (w * iou).sum().item()


class SegEvaluator:
    """
    누적 혼동행렬 기반 Semantic Segmentation evaluator.
    names: ["point_acc", "sem_mAcc", "sem_mIoU", "fwIoU"]
    """
    def __init__(self, num_classes: int, ignore_index: int = -1, device: Optional[torch.device] = None):
        self.num_classes = num_classes
        dev = device if device is not None else torch.device("cpu")
        self.conf = Confusion.empty(num_classes, ignore_index=ignore_index, device=dev)

    @torch.no_grad()
    def add_batch(self, logits: torch.Tensor, target: torch.Tensor):
        """
        logits: (N, C)
        target: (N,)
        """
        pred = logits.argmax(dim=1)
        self.conf.update(pred, target)

    def compute(self, with_per_class: bool = True) -> Dict[str, Any]:
        out = {
            "point_acc": self.conf.point_acc(),
            "sem_mAcc": self.conf.sem_mAcc(),
            "sem_mIoU": self.conf.sem_mIoU(),
            "fwIoU": self.conf.fwIoU(),
        }
        if with_per_class:
            out["per_class_iou"] = self.conf.per_class_iou().cpu().numpy().tolist()
        return out

    def reset(self):
        self.conf.mat.zero_()

    def save_confusion(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savetxt(path, self.conf.mat.cpu().numpy(), fmt="%d", delimiter=",")


# -------------------------
# LR schedule helpers
# -------------------------
def cosine_decay(step: int, total_steps: int, base_lr: float, min_lr: float = 1e-6, warmup_steps: int = 0):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# -------------------------
# Logging helpers
# -------------------------
def log_scalar_dict(d: Dict[str, float]) -> str:
    return " | ".join([f"{k}:{v:.4f}" for k, v in d.items() if isinstance(v, (int, float))])

def dump_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
