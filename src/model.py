import math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from .blocks import (
    CPESABlock, grid_pool, grid_unpool, PoolOut
)

# -----------------------------
# 기본 설정(teacher 1/3 스펙)
# -----------------------------
_DEFAULT_CFG = {
    "input_channels": 6,             # [nx,ny,nz,r,g,b]
    "stages": {
        "dims":   [24, 48, 96, 192], # enc dims (≈ 1/3)
        "blocks": [1,  1,  3,   1 ],
        "heads":  [2,  4,  4,   8 ],
        "mlp_ratio": 3.0,
        "drop_path_max": 0.20,
        "cpe_hidden_ratio": 1.0,
        "pool_stride": [2,2,2,2],    # down×2 per stage
    },
    "bottleneck": {
        "blocks": 1,
        "heads": 8
    },
    "patch": {
        "size": 256,                 # non-overlap patches
    },
    "head": {
        "dropout": 0.0,
        "norm": "LN"
    }
}


def _drop_path_schedule(total: int, dp_max: float) -> List[float]:
    if total <= 1 or dp_max <= 0:
        return [0.0] * total
    return [dp_max * i / (total - 1) for i in range(total)]


def _make_patches(offset: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    offset: (B,) cumulative sizes for concatenated rooms (as in collate)
    returns: (N,) int64 patch index per point (unique across batch)
    Rule: each room slice is chunked by contiguous ranges of size 'patch_size'.
    """
    device = offset.device
    N = int(offset[-1].item())
    patch_ids = torch.empty(N, dtype=torch.long, device=device)
    base_patch = 0
    prev = 0
    for b in range(offset.numel()):
        end = int(offset[b].item())
        n = end - prev
        local = torch.arange(n, device=device, dtype=torch.long) // patch_size
        patch_ids[prev:end] = local + base_patch
        num_patches = int((n + patch_size - 1) // patch_size)
        base_patch += num_patches
        prev = end
    return patch_ids


class Stage(nn.Module):
    """ Encoder/Decoder 공용 Stage: CPESABlock x k """
    def __init__(self, dim, heads, blocks, mlp_ratio, dp_rates, cpe_hidden_ratio, use_ckpt=False):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.blocks = nn.ModuleList([
            CPESABlock(dim, heads, mlp_ratio=mlp_ratio, drop_path=dp_rates[i], cpe_hidden_ratio=cpe_hidden_ratio)
            for i in range(blocks)
        ])

    def forward(self, x: torch.Tensor, coord: torch.Tensor, patch_ids: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                x = checkpoint(lambda _x: blk(_x, coord, patch_ids), x, use_reentrant=False)
            else:
                x = blk(x, coord, patch_ids)
        return x


class PTv3KD(nn.Module):
    """
    U-Net 스타일 4-Stage Encoder + Bottleneck + 4-Stage Decoder.
    - 패치 단위 어텐션(Serialized Attention)
    - 좌표 조건 CPE-lite
    - Grid Pool/Unpool로 다운/업샘플
    - 전부 PyTorch-only (SDPA 사용), 커스텀 CUDA 없음
    """
    def __init__(self, num_classes: int, cfg: Dict[str, Any] = None):
        super().__init__()
        self.cfg = dict(_DEFAULT_CFG)
        if cfg is not None:
            # 얕은 병합(중요 키만 덮어쓰기)
            for k, v in cfg.items():
                if isinstance(v, dict) and k in self.cfg and isinstance(self.cfg[k], dict):
                    self.cfg[k].update(v)
                else:
                    self.cfg[k] = v

        C_in = self.cfg["input_channels"]
        dims  = self.cfg["stages"]["dims"]
        blocks= self.cfg["stages"]["blocks"]
        heads = self.cfg["stages"]["heads"]
        mlp_ratio = self.cfg["stages"]["mlp_ratio"]
        dp_max = self.cfg["stages"]["drop_path_max"]
        cpe_hr = self.cfg["stages"]["cpe_hidden_ratio"]
        strides = self.cfg["stages"]["pool_stride"]
        patch_size = self.cfg["patch"]["size"]

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(C_in, dims[0]),
            nn.LayerNorm(dims[0])
        )

        # DropPath 스케줄 (총 block 수)
        total_blocks = sum(blocks) + self.cfg["bottleneck"]["blocks"] + sum(blocks[::-1])  # enc + bottleneck + dec
        dp_rates = _drop_path_schedule(total_blocks, dp_max)
        it = 0

        # Encoder stages
        self.enc_stages = nn.ModuleList()
        for i in range(4):
            k = blocks[i]
            self.enc_stages.append(
                Stage(dims[i], heads[i], k, mlp_ratio, dp_rates[it:it+k], cpe_hr, use_ckpt=True)
            )
            it += k

        # 🔧 다운프로젝션: 풀링 후 다음 stage 채널로 정렬 (24→48→96→192)
        self.enc_down = nn.ModuleList([
            nn.Identity(),  # enc0 직후는 풀링 다음 enc1로 들어갈 때만 사용하므로 여기선 placeholder
            nn.Sequential(nn.Linear(dims[0], dims[1]), nn.LayerNorm(dims[1])),
            nn.Sequential(nn.Linear(dims[1], dims[2]), nn.LayerNorm(dims[2])),
            nn.Sequential(nn.Linear(dims[2], dims[3]), nn.LayerNorm(dims[3])),
        ])

        # Bottleneck
        self.bottleneck = Stage(dims[-1], self.cfg["bottleneck"]["heads"], self.cfg["bottleneck"]["blocks"],
                                mlp_ratio, dp_rates[it:it+self.cfg["bottleneck"]["blocks"]], cpe_hr, use_ckpt=True)
        it += self.cfg["bottleneck"]["blocks"]

        # Decoder stages (skip concat + 1 block씩 얕게)
        self.dec_proj = nn.ModuleList([
            nn.Linear(dims[3] + dims[3], dims[3]),  # dec3: x3(192) + up3(192) -> 384 -> 192
            nn.Linear(dims[2] + dims[3], dims[2]),  # dec2: x2(96)  + up2(192) -> 288 -> 96
            nn.Linear(dims[1] + dims[2], dims[1]),  # dec1: x1(48)  + up1(96)  -> 144 -> 48
            nn.Linear(dims[0] + dims[1], dims[0]),  # dec0: x0(24)  + up0(48)  -> 72  -> 24
        ])
        self.dec_stages = nn.ModuleList([
            Stage(dims[3], heads[3], 1, mlp_ratio, dp_rates[it:it+1], cpe_hr, use_ckpt=True),
            Stage(dims[2], heads[2], 1, mlp_ratio, dp_rates[it+1:it+2], cpe_hr, use_ckpt=True),
            Stage(dims[1], heads[1], 1, mlp_ratio, dp_rates[it+2:it+3], cpe_hr, use_ckpt=True),
            Stage(dims[0], heads[0], 1, mlp_ratio, dp_rates[it+3:it+4], cpe_hr, use_ckpt=True),
        ])

        # 기타
        self.pool_strides = strides
        self.patch_size = patch_size

        # Head
        if self.cfg["head"]["norm"].upper() == "LN":
            self.head_norm = nn.LayerNorm(dims[0])
        else:
            self.head_norm = nn.Identity()
        self.head_drop = nn.Dropout(self.cfg["head"]["dropout"])
        self.classifier = nn.Linear(dims[0], num_classes)

    @torch.no_grad()
    def _per_stage_patches(self, offset: torch.Tensor, patch_size: int) -> torch.Tensor:
        """현재 구현: 입력 해상도 기준 patch ids (enc0)만 생성."""
        return _make_patches(offset, patch_size)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys:
          coord:  (N,3) int32
          feat:   (N,6) float32  [nx,ny,nz,r,g,b]
          offset: (B,)  int32    cumulative sizes
        returns:
          {"logits": (N, num_classes)}
        """
        coord: torch.Tensor = batch["coord"]
        feat:  torch.Tensor = batch["feat"]
        offset: torch.Tensor = batch["offset"]

        # stem
        x0 = self.stem(feat)  # (N, d0)

        # enc0
        patch_ids0 = self._per_stage_patches(offset.to(torch.long), self.patch_size)
        x0 = self.enc_stages[0](x0, coord, patch_ids0)

        # pool -> enc1
        p0 = grid_pool(coord, x0, stride=self.pool_strides[0])  # (coord1, feat1, mapping)
        coord1, x1 = p0.coord, p0.feat
        x1 = self.enc_down[1](x1)  # 24 -> 48
        patch_ids1 = torch.arange(x1.size(0), device=x1.device) // self.patch_size
        x1 = self.enc_stages[1](x1, coord1, patch_ids1)

        # pool -> enc2
        p1 = grid_pool(coord1, x1, stride=self.pool_strides[1])
        coord2, x2 = p1.coord, p1.feat
        x2 = self.enc_down[2](x2)  # 48 -> 96
        patch_ids2 = torch.arange(x2.size(0), device=x2.device) // self.patch_size
        x2 = self.enc_stages[2](x2, coord2, patch_ids2)

        # pool -> enc3
        p2 = grid_pool(coord2, x2, stride=self.pool_strides[2])
        coord3, x3 = p2.coord, p2.feat
        x3 = self.enc_down[3](x3)  # 96 -> 192
        patch_ids3 = torch.arange(x3.size(0), device=x3.device) // self.patch_size
        x3 = self.enc_stages[3](x3, coord3, patch_ids3)

        # pool -> bottleneck
        p3 = grid_pool(coord3, x3, stride=self.pool_strides[3])
        coord4, x4 = p3.coord, p3.feat
        patch_ids4 = torch.arange(x4.size(0), device=x4.device) // self.patch_size
        x4 = self.bottleneck(x4, coord4, patch_ids4)

        # dec3 (to enc3 res)
        up3 = grid_unpool(PoolOut(coord=coord4, feat=x4, mapping=p3.mapping, parent_ids=p3.parent_ids),
                          child_size=coord3.size(0))
        y3 = torch.cat([x3, up3], dim=1)
        y3 = self.dec_proj[0](y3)
        y3 = self.dec_stages[0](y3, coord3, patch_ids3)

        # dec2 (to enc2 res)
        up2 = grid_unpool(PoolOut(coord=coord3, feat=y3, mapping=p2.mapping, parent_ids=p2.parent_ids),
                          child_size=coord2.size(0))
        y2 = torch.cat([x2, up2], dim=1)
        y2 = self.dec_proj[1](y2)
        y2 = self.dec_stages[1](y2, coord2, patch_ids2)

        # dec1 (to enc1 res)
        up1 = grid_unpool(PoolOut(coord=coord2, feat=y2, mapping=p1.mapping, parent_ids=p1.parent_ids),
                          child_size=coord1.size(0))
        y1 = torch.cat([x1, up1], dim=1)
        y1 = self.dec_proj[2](y1)
        y1 = self.dec_stages[2](y1, coord1, patch_ids1)

        # dec0 (to enc0/input res)
        up0 = grid_unpool(PoolOut(coord=coord1, feat=y1, mapping=p0.mapping, parent_ids=p0.parent_ids),
                          child_size=coord.size(0))
        y0 = torch.cat([x0, up0], dim=1)
        y0 = self.dec_proj[3](y0)
        y0 = self.dec_stages[3](y0, coord, patch_ids0)

        # head
        y0 = self.head_norm(y0)
        y0 = self.head_drop(y0)
        logits = self.classifier(y0)  # (N, num_classes)
        return {"logits": logits}
