# src/dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_FILES = ["coord.npy", "color.npy", "normal.npy", "segment.npy", "instance.npy"]

def _discover_rooms(root: str, split: str):
    split_dir = os.path.join(root, split)
    areas = sorted(glob.glob(os.path.join(split_dir, "Area_*")))
    rooms = []
    for area_dir in areas:
        for room_dir in sorted(glob.glob(os.path.join(area_dir, "*"))):
            if not os.path.isdir(room_dir):
                continue
            files = set(os.listdir(room_dir))
            if all(f in files for f in REQUIRED_FILES):
                rooms.append(room_dir)
    return rooms

class S3DISRoomDataset(Dataset):
    """
    Expects directory structure:
      root/{train,val}/Area_X/<room>/{coord.npy,color.npy,normal.npy,segment.npy,instance.npy}
    Returns dict per sample:
      coord:(N,3) int32, feat:(N,6) float32 [nx,ny,nz,r,g,b], label:(N,) int64, inst:(N,) int64, meta:dict
    """
    def __init__(self, root: str, split: str = "train", strict: bool = True):
        self.root = root
        self.split = split
        self.strict = strict
        self.rooms = _discover_rooms(root, split)
        if len(self.rooms) == 0:
            raise FileNotFoundError(f"No valid rooms under {root}/{split}. "
                                    f"Expected {REQUIRED_FILES} in each room folder.")

    def __len__(self):
        return len(self.rooms)

    def _load_room(self, room_dir: str):
        def npy(name): return np.load(os.path.join(room_dir, name), mmap_mode="r")
        coord   = npy("coord.npy")
        color   = npy("color.npy")
        normal  = npy("normal.npy")     # only correct name supported
        segment = npy("segment.npy")
        inst    = npy("instance.npy")

        N = coord.shape[0]
        if self.strict:
            for k, arr in [("coord",coord),("color",color),("normal",normal),("segment",segment),("instance",inst)]:
                if arr.shape[0] != N:
                    raise ValueError(f"[{room_dir}] length mismatch: {k}={arr.shape[0]} vs coord={N}.")
            if coord.shape[1] != 3 or color.shape[1] != 3 or normal.shape[1] != 3:
                raise ValueError(f"[{room_dir}] coord/color/normal must be (N,3).")
            if segment.ndim != 1 or inst.ndim != 1:
                raise ValueError(f"[{room_dir}] segment/instance must be (N,).")

        coord_t = torch.from_numpy(coord.astype(np.int32, copy=False))
        if color.dtype == np.uint8:
            color_t = torch.from_numpy(color.astype(np.float32) / 255.0)
        else:
            cf = color.astype(np.float32)
            maxv = 255.0 if cf.max() > 1.0 else 1.0
            color_t = torch.from_numpy(cf / maxv)
        normal_t = torch.from_numpy(normal.astype(np.float32, copy=False))
        feat_t = torch.cat([normal_t, color_t], dim=1).contiguous()

        label_t = torch.from_numpy(segment.astype(np.int64, copy=False))
        inst_t  = torch.from_numpy(inst.astype(np.int64, copy=False))

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
