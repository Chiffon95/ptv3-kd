import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Utils
# -----------------------
def _with_default(x, default):
    return default if x is None else x

def segment_mean(x: torch.Tensor, segment_ids: torch.Tensor, num_segments: Optional[int] = None):
    """
    x: (N, C), segment_ids: (N,) int64
    returns: (S, C) where S = num_segments or max(segment)+1
    """
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1 if segment_ids.numel() > 0 else 0
    out = x.new_zeros((num_segments, x.size(1)))
    cnt = x.new_zeros((num_segments, 1))
    out.index_add_(0, segment_ids, x)
    cnt.index_add_(0, segment_ids, torch.ones_like(segment_ids, dtype=x.dtype).unsqueeze(-1))
    cnt = cnt.clamp_min_(1.0)
    return out / cnt

def linear_attention_mask(lengths: torch.Tensor, Lmax: int):
    """
    lengths: (P,) lengths per patch
    returns: attn_mask for SDPA: True where to mask (i.e., invalid positions)
    shape: (P, 1, Lmax, Lmax)
    """
    device = lengths.device
    idx = torch.arange(Lmax, device=device)[None, :]  # (1, Lmax)
    valid = idx < lengths[:, None]                    # (P, Lmax)
    # broadcast to (P, 1, Lmax, Lmax): mask True where invalid either in query or key
    mask = ~(valid[:, None, :, None] & valid[:, None, None, :])
    return mask


# -----------------------
# Regularization bits
# -----------------------
class DropPath(nn.Module):
    """Stochastic Depth: drop residual paths."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep) / keep
        return x * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return self.gamma * x


# -----------------------
# MLP / FFN
# -----------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 3.0, p: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -----------------------
# CPE-lite (coordinate-conditioned)
# -----------------------
class CPELite(nn.Module):
    """
    좌표 대비 패치 중심을 빼서 정규화한 Δp로 작은 MLP를 통과 → 채널에 주입.
    입력:
      x: (N, C)
      coord: (N, 3) int/float
      patch_ids: (N,) 각 포인트가 속한 패치 id (0..P-1)
    """
    def __init__(self, dim: int, hidden_ratio: float = 1.0, dwpointwise: bool = True):
        super().__init__()
        hid = max(8, int(dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(3, hid), nn.GELU(),
            nn.Linear(hid, dim)
        )
        self.dwpointwise = dwpointwise
        if dwpointwise:
            self.dw = nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=True)
            self.pw = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, coord: torch.Tensor, patch_ids: torch.Tensor):
        # 패치 중심
        patch_centers = segment_mean(coord.to(x.dtype), patch_ids, int(patch_ids.max().item())+1)
        delta = coord.to(x.dtype) - patch_centers[patch_ids]  # (N,3)
        pos = self.mlp(delta)  # (N,C)
        out = x + pos
        if self.dwpointwise:
            # depthwise over tokens as channelwise 1d conv
            y = self.dw(out.transpose(0,1).unsqueeze(0)).squeeze(0).transpose(0,1)  # (N,C)
            out = self.pw(y)
        return self.ln(out)

# src/blocks.py 의 SerializedAttention 정의를 이걸로 교체
class SerializedAttention(nn.Module):
    """
    메모리 세이프 버전:
    - 패딩/거대 마스크 없이 패치별로 SDPA 호출
    - 장점: OOM 방지
    - 단점: 약간 느릴 수 있으나 5080이면 충분
    """
    def __init__(self, dim: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.h = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ln = nn.LayerNorm(dim)

    @torch.no_grad()
    def _sort_and_ranges(self, patch_ids: torch.Tensor):
        idx = torch.argsort(patch_ids)
        p_sorted = patch_ids.index_select(0, idx)
        _, counts = torch.unique_consecutive(p_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, device=counts.device, dtype=counts.dtype), counts[:-1]]), dim=0)
        return idx, counts.tolist(), starts.tolist()

    def forward(self, x: torch.Tensor, patch_ids: torch.Tensor):
        x_in = x
        x = self.ln(x)

        # 정렬 & 범위 계산
        idx = torch.argsort(patch_ids)
        p_sorted = patch_ids.index_select(0, idx)
        _, counts = torch.unique_consecutive(p_sorted, return_counts=True)
        starts = torch.cumsum(torch.cat([torch.zeros(1, device=counts.device, dtype=counts.dtype), counts[:-1]]), dim=0)

        x_sorted = x.index_select(0, idx)           # (N,C)
        C = self.dim
        Dh = C // self.h

        # 🔧 최종 출력 텐서(원래 순서) 미리 할당 → scatter로 채움 (추가 복제 방지)
        out = None

        # 🔧 패치별 루프: 패치마다 qkv 계산 (거대한 qkv_all 제거)
        for L, st in zip(counts.tolist(), starts.tolist()):
            if L == 0:
                continue
            sl = slice(st, st + L)
            xs = x_sorted[sl]                       # (L,C)

            qkv = self.qkv(xs)                      # (L,3C)  ← 패치 단위로만 계산
            q, k, v = qkv.chunk(3, dim=-1)          # each (L,C)

            # (L,C) -> (H,L,Dh)
            q = q.view(L, self.h, Dh).transpose(0, 1)
            k = k.view(L, self.h, Dh).transpose(0, 1)
            v = v.view(L, self.h, Dh).transpose(0, 1)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)  # (H,L,Dh)
            y = y.transpose(0, 1).contiguous().view(L, C)  # (L,C)

            # 처음 한 번만, y의 dtype으로 버퍼 생성 (AMP면 fp16)
            if out is None:
                out = x.new_empty(x.shape, dtype=y.dtype, device=y.device)

            orig_pos = idx[sl]
            out.index_copy_(0, orig_pos, y)

        out = self.proj_drop(self.proj(out))
        return x_in + out


# -----------------------
# Transformer-like Block
# -----------------------
class SATransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 3.0, drop_path: float = 0.0, qkv_bias: bool = True):
        super().__init__()
        self.attn = SerializedAttention(dim, heads, qkv_bias=qkv_bias)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma1 = LayerScale(dim, 1e-5)
        self.gamma2 = LayerScale(dim, 1e-5)

    def forward(self, x: torch.Tensor, patch_ids: torch.Tensor):
        x = x + self.drop_path(self.gamma1(self.attn(x, patch_ids)))
        x = x + self.drop_path(self.gamma2(self.mlp(self.mlp_norm(x))))
        return x


# -----------------------
# Grid Pool / Unpool
# -----------------------
@dataclass
class PoolOut:
    coord: torch.Tensor      # (M,3) int32
    feat: torch.Tensor       # (M,C)
    mapping: torch.Tensor    # (N,) indices of parent per point (for unpool)
    parent_ids: torch.Tensor # (M,) 0..M-1

def grid_pool(coord: torch.Tensor, feat: torch.Tensor, stride: int = 2) -> PoolOut:
    """
    coord: (N,3) int32, feat: (N,C)
    정수 그리드에서 stride로 다운샘플. parent = floor(coord/stride)
    """
    parent = torch.div(coord, stride, rounding_mode='floor')
    # 해시 (x,y,z) -> scalar key
    device = coord.device
    parent64 = parent.to(torch.int64)
    key = (parent64[:, 0] * 73856093) ^ (parent64[:, 1] * 19349663) ^ (parent64[:, 2] * 83492791)
    uniq, inv = torch.unique(key, return_inverse=True)
    M = uniq.numel()
    # 좌표 평균(정수 반올림) + 피처 평균
    parent_coord = segment_mean(parent.to(feat.dtype), inv, M).round().to(torch.int32)
    parent_feat = segment_mean(feat, inv, M)
    return PoolOut(coord=parent_coord, feat=parent_feat, mapping=inv, parent_ids=torch.arange(M, device=device, dtype=torch.long))

def grid_unpool(parent_out: PoolOut, child_size: int) -> torch.Tensor:
    """
    parent_out.mapping: (N_child,) -> parent index
    returns gathered parent_feat to child points: (N_child, C)
    """
    return parent_out.feat[parent_out.mapping]


# -----------------------
# CPE + SA Block wrapper (편의)
# -----------------------
class CPESABlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 3.0, drop_path: float = 0.0, cpe_hidden_ratio: float = 1.0):
        super().__init__()
        self.cpe = CPELite(dim, hidden_ratio=cpe_hidden_ratio, dwpointwise=True)
        self.sa = SATransformerBlock(dim, heads, mlp_ratio, drop_path)

    def forward(self, x: torch.Tensor, coord: torch.Tensor, patch_ids: torch.Tensor):
        x = self.cpe(x, coord, patch_ids)
        x = self.sa(x, patch_ids)
        return x
