#!/usr/bin/env python3
# scripts/verify_s3dis.py
import os, sys, csv, argparse
import numpy as np

REQUIRED = ["coord.npy","color.npy","normal.npy","segment.npy","instance.npy"]

def shape_str(a):
    try: return "x".join(map(str, a.shape))
    except: return "-"

def dtype_str(a):
    try: return str(a.dtype)
    except: return "-"

def load_npy(path):
    try:
        return np.load(path, mmap_mode="r"), None
    except Exception as e:
        return None, f"ERROR: {e}"

def scan_room(room_dir, num_classes=None):
    info = {
        "room_dir": room_dir,
        "missing": [],
        "errors": [],
        "length_mismatch": False,
        "segment_is_column": False,
        "instance_is_column": False,
        "shapes": {},
        "dtypes": {},
        "lengths": {},
        "label_range_ok": None,
        "segment_min": None,
        "segment_max": None,
    }
    arrays = {}
    for fn in REQUIRED:
        fp = os.path.join(room_dir, fn)
        if not os.path.isfile(fp):
            info["missing"].append(fn); continue
        arr, err = load_npy(fp)
        if err:
            info["errors"].append(f"{fn}: {err}"); continue
        arrays[fn] = arr
        info["shapes"][fn] = shape_str(arr)
        info["dtypes"][fn] = dtype_str(arr)

    # Quick checks
    if "coord.npy" in arrays:
        N = arrays["coord.npy"].shape[0]
    elif arrays:
        # fallback to any present
        any_arr = next(iter(arrays.values()))
        N = any_arr.shape[0] if hasattr(any_arr, "shape") and len(any_arr.shape) > 0 else None
    else:
        info["errors"].append("No arrays loaded"); return info

    # (N,1) → 플래그만 세움(고치진 않음)
    for key, flag_key in [("segment.npy","segment_is_column"), ("instance.npy","instance_is_column")]:
        if key in arrays:
            a = arrays[key]
            if a.ndim == 2 and a.shape[1] == 1:
                info[flag_key] = True

    # length mismatch
    for fn, arr in arrays.items():
        if hasattr(arr, "shape") and len(arr.shape) >= 1 and N is not None:
            if arr.shape[0] != N:
                info["length_mismatch"] = True
                info["errors"].append(f"LENGTH MISMATCH: {fn} has {arr.shape[0]} vs coord {N}")
        info["lengths"][fn] = arr.shape[0] if hasattr(arr,"shape") and len(arr.shape)>0 else None

    # semantic label range check (optional)
    if num_classes is not None and "segment.npy" in arrays:
        seg = arrays["segment.npy"]
        try:
            if seg.ndim == 2 and seg.shape[1] == 1:
                seg = seg.reshape(-1)
            seg_min = int(seg.min())
            seg_max = int(seg.max())
            info["segment_min"] = seg_min
            info["segment_max"] = seg_max
            info["label_range_ok"] = (0 <= seg_min) and (seg_max < num_classes)
            if not info["label_range_ok"]:
                info["errors"].append(f"SEGMENT OUT OF RANGE: [{seg_min}, {seg_max}] not in [0, {num_classes-1}]")
        except Exception as e:
            info["errors"].append(f"segment range check failed: {e}")

    return info

def main():
    ap = argparse.ArgumentParser(description="Verify S3DIS preprocessed dataset structure")
    ap.add_argument("--root", type=str, required=True, help="dataset root (e.g., ./data/s3dis)")
    ap.add_argument("--csv", type=str, default="verify_report.csv", help="save CSV report path")
    ap.add_argument("--num-classes", type=int, default=None, help="optional, check segment label range [0,C-1]")
    args = ap.parse_args()

    root = args.root
    splits = ["train","val"]
    rooms = []
    for sp in splits:
        sp_dir = os.path.join(root, sp)
        if not os.path.isdir(sp_dir): 
            print(f"[warn] split missing: {sp_dir}")
            continue
        for area in sorted(os.listdir(sp_dir)):
            area_dir = os.path.join(sp_dir, area)
            if not (os.path.isdir(area_dir) and area.startswith("Area_")):
                continue
            for room in sorted(os.listdir(area_dir)):
                room_dir = os.path.join(area_dir, room)
                if os.path.isdir(room_dir):
                    rooms.append((sp, area, room, room_dir))

    if not rooms:
        print("[err] no rooms found under", root)
        sys.exit(1)

    total = len(rooms)
    bad_missing = 0
    bad_errors = 0
    bad_len = 0
    col_count_seg = 0
    col_count_ins = 0
    bad_label_range = 0

    rows = []
    for sp, area, room, room_dir in rooms:
        info = scan_room(room_dir, num_classes=args.num_classes)
        # counters
        if info["missing"]: bad_missing += 1
        if info["errors"]: bad_errors += 1
        if info["length_mismatch"]: bad_len += 1
        if info["segment_is_column"]: col_count_seg += 1
        if info["instance_is_column"]: col_count_ins += 1
        if info["label_range_ok"] is False: bad_label_range += 1

        rows.append({
            "split": sp,
            "area": area,
            "room": room,
            "room_dir": room_dir,
            "missing": ";".join(info["missing"]) if info["missing"] else "",
            "errors": ";".join(info["errors"]) if info["errors"] else "",
            "len_mismatch": info["length_mismatch"],
            "segment_is_(N,1)": info["segment_is_column"],
            "instance_is_(N,1)": info["instance_is_column"],
            "coord.shape": info["shapes"].get("coord.npy","-"),
            "color.shape": info["shapes"].get("color.npy","-"),
            "normal.shape": info["shapes"].get("normal.npy","-"),
            "segment.shape": info["shapes"].get("segment.npy","-"),
            "instance.shape": info["shapes"].get("instance.npy","-"),
            "coord.dtype": info["dtypes"].get("coord.npy","-"),
            "color.dtype": info["dtypes"].get("color.npy","-"),
            "normal.dtype": info["dtypes"].get("normal.npy","-"),
            "segment.dtype": info["dtypes"].get("segment.npy","-"),
            "instance.dtype": info["dtypes"].get("instance.npy","-"),
            "segment_min": info["segment_min"],
            "segment_max": info["segment_max"],
            "label_range_ok": info["label_range_ok"],
        })

    # summary
    print(f"\n=== SUMMARY ===")
    print(f"Rooms scanned       : {total}")
    print(f"Rooms missing files : {bad_missing}")
    print(f"Rooms with errors   : {bad_errors}")
    print(f"Length mismatches   : {bad_len}")
    print(f"segment (N,1) rooms : {col_count_seg}")
    print(f"instance (N,1) rooms: {col_count_ins}")
    if args.num_classes is not None:
        print(f"Label-range invalid : {bad_label_range} (out of [0,{args.num_classes-1}])")

    # write CSV
    out = args.csv
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nReport saved → {out}")
    print("Tip: open in VS Code / Excel and filter by columns like 'missing', 'errors', 'segment_is_(N,1)'.")
    
if __name__ == "__main__":
    main()
