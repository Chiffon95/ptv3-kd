# src/dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_FILES = ["coord.npy", "color.npy", "normal.npy", "segment.npy", "instance.npy"]

def _discover_rooms(root: str, split: str, areas=None):
    split_dir = root if split == "" else os.path.join(root, split)

    if areas is None or len(areas) == 0:
        area_globs = [os.path.join(split_dir, "Area_*")]
    else:
        area_globs = [os.path.join(split_dir, a) for a in areas]

    rooms = []

    for ag in area_globs:
        for area_dir in sorted(glob.glob(ag)):
            if not os.path.isdir(area_dir):
                continue
            for room_dir in sorted(glob.glob(os.path.join(area_dir, "*"))):
                if not os.path.isdir(room_dir):
                    continue
                files = set(os.listdir(room_dir))
                if all(f in files for f in REQUIRED_FILES):
                    rooms.append(room_dir)
    return rooms

class S3DISRoomDataset(Dataset):
    """
    Expects:
      root/{train,val}/Area_X/<room>/{coord.npy,color.npy,normal.npy,segment.npy,instance.npy}
    Returns dict per sample:
      coord:(N,3) int32, feat:(N,6) float32 [nx,ny,nz,r,g,b], label:(N,) int64, inst:(N,) int64, meta:dict
    """
    def __init__(self, root: str, split: str = "train", strict: bool = True, areas=None):
        self.root = root
        self.split = split
        self.strict = strict
        self.rooms = _discover_rooms(root, split, areas=areas)
        if len(self.rooms) == 0:
            msg = f"No valid rooms under {root}/{split}"
            if areas: msg += f" with areas={areas}"
            msg += f". Expected {REQUIRED_FILES} in each room folder."
            raise FileNotFoundError(msg)

    def __len__(self):
        return len(self.rooms)

    def _load_room(self, room_dir: str):
        # mmap_mode=None 로 두면 writable 복사본이 되어 경고 제거에 도움이 됨
        def npy(name): return np.load(os.path.join(room_dir, name), mmap_mode=None)
        coord   = npy("coord.npy")
        color   = npy("color.npy")
        normal  = npy("normal.npy")
        segment = npy("segment.npy")
        inst    = npy("instance.npy")

        # (N,1) -> (N,)
        def _squeeze_vec(a):
            if a.ndim == 2 and a.shape[1] == 1:
                return a.reshape(-1)
            return a
        segment = _squeeze_vec(segment)
        inst    = _squeeze_vec(inst)

        N = coord.shape[0]
        if self.strict:
            for k, arr in [("coord",coord),("color",color),("normal",normal),("segment",segment),("instance",inst)]:
                if arr.shape[0] != N:
                    raise ValueError(f"[{room_dir}] length mismatch: {k}={arr.shape[0]} vs coord={N}.")
            if coord.shape[1] != 3 or color.shape[1] != 3 or normal.shape[1] != 3:
                raise ValueError(f"[{room_dir}] coord/color/normal must be (N,3).")
            if segment.ndim != 1 or inst.ndim != 1:
                raise ValueError(f"[{room_dir}] segment/instance must be (N,).")

        # to tensors (copy=True 로 read-only 경고 방지)
        coord_t = torch.from_numpy(coord.astype(np.int32,  copy=True))
        if color.dtype == np.uint8:
            color_t = torch.from_numpy((color.astype(np.float32, copy=False) / 255.0).copy())
        else:
            cf = color.astype(np.float32, copy=False)
            maxv = 255.0 if cf.max() > 1.0 else 1.0
            color_t = torch.from_numpy((cf / maxv).copy())
        normal_t = torch.from_numpy(normal.astype(np.float32, copy=True))
        feat_t = torch.cat([normal_t, color_t], dim=1).contiguous()

        label_t = torch.from_numpy(segment.astype(np.int64, copy=True))
        inst_t  = torch.from_numpy(inst.astype(np.int64, copy=True))

        meta = {
            "area": os.path.basename(os.path.dirname(room_dir)),
            "room": os.path.basename(room_dir),
            "path": room_dir,
        }
        return {"coord": coord_t, "feat": feat_t, "label": label_t, "inst": inst_t, "meta": meta}

    def __getitem__(self, idx: int):
        return self._load_room(self.rooms[idx])

def s3dis_collate_fn(batch):
    coords, feats, labels, insts, metas, sizes = [], [], [], [], [], []
    for sample in batch:
        coords.append(sample["coord"])
        feats.append(sample["feat"])
        labels.append(sample["label"])
        insts.append(sample["inst"])
        metas.append(sample["meta"])
        sizes.append(sample["coord"].shape[0])
    coord = torch.cat(coords, dim=0)
    feat  = torch.cat(feats, dim=0)
    label = torch.cat(labels, dim=0)
    inst  = torch.cat(insts, dim=0)
    offset = torch.tensor(np.cumsum(sizes), dtype=torch.int32)
    return {"coord": coord, "feat": feat, "label": label, "inst": inst, "offset": offset, "meta": metas}
