"""
Generate synthetic occlusion images via CopyPaste augmentation.

Reads the converted flower_amodal dataset and creates new training images
by pasting one flower on top of another, producing realistic synthetic
occlusion with correct amodal / visible / occluded annotations.

This runs ONCE before training as a data preparation step.
Output is a self-contained dataset that convert_flower_dataset.py already
understands — just point FLOWER_DATA_ROOT at the output directory.

Algorithm per synthetic sample
-------------------------------
1. Pick image A  (background flower — will become occluded)
2. Pick image B  (foreground flower — pasted on top)
3. Centre flower B over flower A with a random offset so overlap is 20–60 %
4. Composite:  result[B_mask] = imgB[B_mask_origin]
5. Annotations
   • Flower A (occluded):
       amodal_segm   = original amodal  (shape unchanged)
       visible_segm  = amodal  AND NOT  B_mask_on_A
       occluder_segm = amodal  AND      B_mask_on_A
   • Flower B (occluder, fully visible):
       amodal_segm   = B_mask at new position
       visible_segm  = same  (nothing in front)
       occluder_segm = empty

Usage
-----
# Step 1 — generate (run from anywhere, no detectron2 needed)
python generate_synthetic_occlusions.py \
    --src  ~/datasets/flower_amodal \
    --dst  ~/datasets/flower_amodal_aug \
    --multiplier 2          # produce 2× extra images per training image
    --seed 0

# Step 2 — train using the augmented dataset
FLOWER_DATA_ROOT=~/datasets/flower_amodal_aug \
python train_net.py ...
"""

import argparse
import copy
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util


# ── RLE helpers ───────────────────────────────────────────────────────────────

def decode(segm, h, w):
    if segm is None:
        return np.zeros((h, w), dtype=bool)
    if isinstance(segm, dict):
        counts = segm["counts"]
        if isinstance(counts, str):
            counts = counts.encode("utf-8")
        return mask_util.decode({"size": segm["size"], "counts": counts}).astype(bool)
    return np.zeros((h, w), dtype=bool)


def encode(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y1, y2 = int(np.where(rows)[0][[0, -1]])
    x1, x2 = int(np.where(cols)[0][[0, -1]])
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


# ── paste helper ──────────────────────────────────────────────────────────────

def paste_flower(img_a, mask_a_amodal,
                 img_b, mask_b_amodal, mask_b_visible,
                 rng, target_overlap=(0.20, 0.60)):
    """
    Paste flower B onto image A so the masks overlap by target_overlap fraction
    of flower A's area.  Returns (composited_image, b_amodal_at_new_pos,
    b_visible_at_new_pos) or None if a valid position cannot be found.
    """
    H, W = img_a.shape[:2]

    # bounding boxes of both flowers
    r_a = mask_bbox(mask_a_amodal)   # [x,y,w,h]
    r_b = mask_bbox(mask_b_amodal)
    if r_a[2] == 0 or r_b[2] == 0:
        return None

    cx_a = r_a[0] + r_a[2] // 2
    cy_a = r_a[1] + r_a[3] // 2

    # try random offsets until we hit the target overlap range
    for _ in range(40):
        dx = rng.randint(-r_a[2] // 2, r_a[2] // 2)
        dy = rng.randint(-r_a[3] // 2, r_a[3] // 2)

        # shift: move the centre of B to (cx_a+dx, cy_a+dy)
        cx_b = r_b[0] + r_b[2] // 2
        cy_b = r_b[1] + r_b[3] // 2
        shift_x = (cx_a + dx) - cx_b
        shift_y = (cy_a + dy) - cy_b

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        b_amod = cv2.warpAffine(mask_b_amodal.astype(np.uint8), M, (W, H),
                                flags=cv2.INTER_NEAREST).astype(bool)
        b_vis  = cv2.warpAffine(mask_b_visible.astype(np.uint8), M, (W, H),
                                flags=cv2.INTER_NEAREST).astype(bool)

        overlap = float((b_amod & mask_a_amodal).sum())
        if mask_a_amodal.sum() == 0:
            break
        frac = overlap / float(mask_a_amodal.sum())
        if target_overlap[0] <= frac <= target_overlap[1]:
            # composite
            img_b_shifted = cv2.warpAffine(img_b, M, (W, H))
            composite = img_a.copy()
            composite[b_vis] = img_b_shifted[b_vis]
            return composite, b_amod, b_vis

    return None


# ── annotation builder ────────────────────────────────────────────────────────

def build_annotations(ann_a, ann_b_template,
                      mask_a_amodal, mask_a_visible,
                      mask_b_amodal_new, mask_b_visible_new,
                      next_ann_id, cat_b):
    """
    Build two annotations for a synthetic image:
      ann_occluded  — flower A with updated visible / occluder masks
      ann_occluder  — flower B (pasted, fully visible)
    """
    # occluded region = where B covers A's amodal shape
    b_covers_a   = mask_b_amodal_new & mask_a_amodal
    visible_a    = mask_a_amodal & ~b_covers_a

    ann_occluded = {
        "id":            next_ann_id,
        "image_id":      None,           # filled by caller
        "category_id":   ann_a["category_id"],
        "iscrowd":       0,
        "amodal_segm":   encode(mask_a_amodal),
        "visible_segm":  encode(visible_a),
        "occluder_segm": encode(b_covers_a),
        "amodal_bbox":   mask_bbox(mask_a_amodal),
        "visible_bbox":  mask_bbox(visible_a),
        "bbox":          mask_bbox(visible_a),
        "area":          int(mask_a_amodal.sum()),
        "occlusion_fraction": float(b_covers_a.sum()) / max(1, mask_a_amodal.sum()),
    }

    ann_occluder = {
        "id":            next_ann_id + 1,
        "image_id":      None,
        "category_id":   cat_b,
        "iscrowd":       0,
        "amodal_segm":   encode(mask_b_amodal_new),
        "visible_segm":  encode(mask_b_visible_new),
        "occluder_segm": encode(np.zeros_like(mask_b_amodal_new)),
        "amodal_bbox":   mask_bbox(mask_b_amodal_new),
        "visible_bbox":  mask_bbox(mask_b_visible_new),
        "bbox":          mask_bbox(mask_b_visible_new),
        "area":          int(mask_b_amodal_new.sum()),
        "occlusion_fraction": 0.0,
    }

    return ann_occluded, ann_occluder


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",        required=True,
                        help="Converted dataset root (~/datasets/flower_amodal)")
    parser.add_argument("--dst",        required=True,
                        help="Output augmented dataset root")
    parser.add_argument("--multiplier", type=int,   default=2,
                        help="Synthetic images per original training image (default 2)")
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--overlap-min", type=float, default=0.20)
    parser.add_argument("--overlap-max", type=float, default=0.60)
    args = parser.parse_args()

    src  = Path(os.path.expanduser(args.src))
    dst  = Path(os.path.expanduser(args.dst))
    rng  = random.Random(args.seed)

    # ── copy val and test splits unchanged ────────────────────────────────────
    print("Copying val and test splits …")
    for split in ("val", "test"):
        for subdir in ("annotations", f"{split}/images"):
            src_p = src / subdir
            dst_p = dst / subdir
            dst_p.mkdir(parents=True, exist_ok=True)
        # annotations
        for jf in (f"{split}_amodal.json",):
            s = src / "annotations" / jf
            d = dst / "annotations" / jf
            if s.exists() and not d.exists():
                shutil.copy2(s, d)
        # images (symlinks)
        img_src = src / split / "images"
        img_dst = dst / split / "images"
        if img_src.exists():
            for f in img_src.iterdir():
                link = img_dst / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())

    # ── load train split ──────────────────────────────────────────────────────
    train_json = src / "annotations" / "train_amodal.json"
    print(f"Loading {train_json} …")
    with open(train_json) as f:
        data = json.load(f)

    id2img = {img["id"]: img for img in data["images"]}
    id2ann = {ann["image_id"]: ann for ann in data["annotations"]}  # 1 ann / image

    img_dir_src = src / "train" / "images"
    img_dir_dst = dst / "train" / "images"
    img_dir_dst.mkdir(parents=True, exist_ok=True)
    ann_dir_dst = dst / "annotations"
    ann_dir_dst.mkdir(parents=True, exist_ok=True)

    # ── start building output JSON ────────────────────────────────────────────
    out_images      = copy.deepcopy(data["images"])
    out_annotations = copy.deepcopy(data["annotations"])

    # symlink all original training images
    print("Symlinking original training images …")
    for img_meta in data["images"]:
        src_f = img_dir_src / img_meta["file_name"]
        dst_f = img_dir_dst / img_meta["file_name"]
        if src_f.exists() and not dst_f.exists():
            dst_f.symlink_to(src_f.resolve())

    next_img_id = max(img["id"] for img in data["images"]) + 1
    next_ann_id = max(ann["id"] for ann in data["annotations"]) + 1

    image_ids = list(id2img.keys())
    total     = len(image_ids) * args.multiplier
    done      = 0

    print(f"\nGenerating {total} synthetic images "
          f"(multiplier={args.multiplier}, "
          f"overlap={args.overlap_min:.0%}–{args.overlap_max:.0%}) …\n")

    for img_id_a in image_ids:
        img_meta_a = id2img[img_id_a]
        ann_a      = id2ann.get(img_id_a)
        if ann_a is None:
            continue

        for _ in range(args.multiplier):
            # pick a different image as the paste source
            img_id_b = rng.choice(image_ids)
            while img_id_b == img_id_a:
                img_id_b = rng.choice(image_ids)

            img_meta_b = id2img[img_id_b]
            ann_b      = id2ann.get(img_id_b)
            if ann_b is None:
                continue

            # load images
            path_a = img_dir_src / img_meta_a["file_name"]
            path_b = img_dir_src / img_meta_b["file_name"]
            img_a  = cv2.imread(str(path_a))
            img_b  = cv2.imread(str(path_b))
            if img_a is None or img_b is None:
                continue

            H_a, W_a = img_a.shape[:2]
            H_b, W_b = img_b.shape[:2]

            # decode masks
            mask_a_amod = decode(ann_a.get("amodal_segm"),   H_a, W_a)
            mask_a_vis  = decode(ann_a.get("visible_segm"),  H_a, W_a)
            mask_b_amod = decode(ann_b.get("amodal_segm"),   H_b, W_b)
            mask_b_vis  = decode(ann_b.get("visible_segm"),  H_b, W_b)

            # resize B to match A if sizes differ
            if (H_b, W_b) != (H_a, W_a):
                img_b       = cv2.resize(img_b,       (W_a, H_a))
                mask_b_amod = cv2.resize(mask_b_amod.astype(np.uint8),
                                         (W_a, H_a), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_b_vis  = cv2.resize(mask_b_vis.astype(np.uint8),
                                         (W_a, H_a), interpolation=cv2.INTER_NEAREST).astype(bool)

            result = paste_flower(
                img_a, mask_a_amod,
                img_b, mask_b_amod, mask_b_vis,
                rng,
                target_overlap=(args.overlap_min, args.overlap_max),
            )
            if result is None:
                continue

            composite, b_amod_new, b_vis_new = result

            # build new filename
            fname_syn = f"syn_{next_img_id:06d}.jpg"
            cv2.imwrite(str(img_dir_dst / fname_syn), composite,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            # image record
            out_images.append({
                "id":        next_img_id,
                "file_name": fname_syn,
                "width":     W_a,
                "height":    H_a,
            })

            # annotations
            ann_occ, ann_occ_r = build_annotations(
                ann_a, ann_b,
                mask_a_amod, mask_a_vis,
                b_amod_new, b_vis_new,
                next_ann_id,
                cat_b=ann_b["category_id"],
            )
            ann_occ["image_id"]   = next_img_id
            ann_occ_r["image_id"] = next_img_id
            out_annotations.append(ann_occ)
            out_annotations.append(ann_occ_r)

            next_img_id += 1
            next_ann_id += 2
            done += 1

            if done % 500 == 0 or done == total:
                print(f"  {done}/{total} synthetic images created")

    # ── write augmented train JSON ────────────────────────────────────────────
    out_json = {
        "info":        data.get("info", {}),
        "categories":  data["categories"],
        "images":      out_images,
        "annotations": out_annotations,
    }
    out_path = ann_dir_dst / "train_amodal.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f)

    orig  = len(data["images"])
    synt  = done
    total_imgs = len(out_images)
    print(f"\nDone.")
    print(f"  Original training images : {orig}")
    print(f"  Synthetic images added   : {synt}")
    print(f"  Total training images    : {total_imgs}")
    print(f"  Output dataset           : {dst}")
    print(f"\nTo train with augmented dataset:")
    print(f"  FLOWER_DATA_ROOT={dst} python train_net.py ...")


if __name__ == "__main__":
    main()
