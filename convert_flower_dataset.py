"""
Convert a raw flower dataset → flower_aistron format (aistron-compatible COCO-like).

Expected source annotation format (standard COCO + amodal fields):
  Each annotation must have:
    "segmentation"         — visible (modal) mask
    "amodal_segmentation"  — full amodal mask (RLE)

Output layout:
  <dst>/
    annotations/
      train_amodal.json
      val_amodal.json
    train/images/
    val/images/

Usage (symlinks — fast, keeps source untouched):
  python convert_flower_dataset.py \
      --src /mnt/d/FlowerDataset \
      --dst /mnt/d/flower_aistron

Usage (copy images to Linux ext4 — recommended for WSL2 training speed):
  python convert_flower_dataset.py \
      --src /mnt/d/FlowerDataset \
      --dst ~/datasets/flower_aistron \
      --copy-images
"""

import argparse
import json
import os
import pathlib
import shutil
import pycocotools.mask as mask_util


def compute_amodal_bbox(rle_mask):
    return [float(v) for v in mask_util.toBbox(rle_mask)]


def convert_annotations(src_json: str) -> tuple:
    with open(src_json) as f:
        data = json.load(f)

    new_annotations, skipped = [], 0
    for ann in data["annotations"]:
        vis_segm  = ann.get("segmentation")
        amod_segm = ann.get("amodal_segmentation")
        if vis_segm is None or amod_segm is None:
            skipped += 1
            continue
        new_ann = {
            "id":           ann["id"],
            "image_id":     ann["image_id"],
            "category_id":  ann["category_id"],
            "iscrowd":      ann.get("iscrowd", 0),
            "area":         ann["area"],
            "visible_segm": vis_segm,
            "amodal_segm":  amod_segm,
            "visible_bbox": ann["bbox"],
            "amodal_bbox":  compute_amodal_bbox(amod_segm),
            "bbox":         ann["bbox"],
        }
        for extra in ("occlusion_fraction", "occlusion_side"):
            if extra in ann:
                new_ann[extra] = ann[extra]
        new_annotations.append(new_ann)

    print(f"  Converted {len(new_annotations)} annotations, skipped {skipped}")
    return data["images"], data["categories"], new_annotations, data.get("info", {})


def make_image_links(images, src_image_dir, dst_image_dir, copy: bool):
    dst = pathlib.Path(dst_image_dir)
    dst.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(src_image_dir)
    missing = 0
    for img in images:
        src_file = src / img["file_name"]
        dst_file = dst / img["file_name"]
        if dst_file.exists() or dst_file.is_symlink():
            continue
        if src_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copy2(src_file, dst_file)
            else:
                dst_file.symlink_to(src_file.resolve())
        else:
            missing += 1
    if missing:
        print(f"  WARNING: {missing} source images not found")


def convert_split(split, src_json, src_image_dir, dst_root, copy):
    print(f"\n--- {split} ---")
    images, categories, annotations, info = convert_annotations(src_json)

    ann_dir  = os.path.join(dst_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    out_json = os.path.join(ann_dir, f"{split}_amodal.json")
    with open(out_json, "w") as f:
        json.dump({"info": info, "categories": categories,
                   "images": images, "annotations": annotations}, f)
    print(f"  Wrote {out_json}")

    img_dir = os.path.join(dst_root, split, "images")
    print(f"  {'Copying' if copy else 'Symlinking'} {len(images)} images → {img_dir}")
    make_image_links(images, src_image_dir, img_dir, copy=copy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Source dataset root (must contain annotations/ and images/)")
    parser.add_argument("--dst", default="/mnt/d/flower_aistron",
                        help="Output dataset root")
    parser.add_argument("--train-json", default=None,
                        help="Path to train annotation JSON (default: <src>/annotations/flower_train.json)")
    parser.add_argument("--val-json", default=None,
                        help="Path to val annotation JSON (default: <src>/annotations/flower_val.json)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking (recommended for WSL2 ext4)")
    args = parser.parse_args()

    src_images = os.path.join(args.src, "images")
    train_json = args.train_json or os.path.join(args.src, "annotations", "flower_train.json")
    val_json   = args.val_json   or os.path.join(args.src, "annotations", "flower_val.json")

    convert_split("train", train_json, src_images, args.dst, args.copy_images)
    convert_split("val",   val_json,   src_images, args.dst, args.copy_images)

    print(f"\nDone. Dataset at: {args.dst}")
    print(f"Set FLOWER_DATA_ROOT={args.dst} when training.")


if __name__ == "__main__":
    main()
