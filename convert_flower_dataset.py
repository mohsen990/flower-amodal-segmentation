"""
Convert flowers_amodal_output dataset → aistron-compatible format.

Source annotation fields (per annotation):
  "segmentation"           — visible mask (RLE)
  "amodal_segmentation"    — full amodal mask (RLE)
  "occluded_segmentation"  — occluded (hidden) part mask (RLE)

Output layout:
  <dst>/
    annotations/
      train_amodal.json
      val_amodal.json
      test_amodal.json
    train/images/  (symlinks or copies)
    val/images/
    test/images/

Usage (symlinks — fast, keeps source on Windows NTFS):
  python convert_flower_dataset.py \
      --src /mnt/e/Dataset/flowers_amodal_output \
      --dst ~/datasets/flower_amodal

Usage (copy to Linux ext4 — recommended for WSL2 training speed):
  python convert_flower_dataset.py \
      --src /mnt/e/Dataset/flowers_amodal_output \
      --dst ~/datasets/flower_amodal \
      --copy-images
"""

import argparse
import json
import os
import pathlib
import shutil
import pycocotools.mask as mask_util


def compute_bbox_from_rle(rle_mask):
    """Return [x, y, w, h] bbox from an RLE mask dict."""
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
            "amodal_bbox":  compute_bbox_from_rle(amod_segm),
            "bbox":         compute_bbox_from_rle(amod_segm),
        }

        # occluded part mask → stored as occluder_segm (used by background_objs_segm slot)
        occ_segm = ann.get("occluded_segmentation")
        if occ_segm is not None:
            new_ann["occluder_segm"] = occ_segm

        for extra in ("occlusion_fraction", "occlusion_side"):
            if extra in ann:
                new_ann[extra] = ann[extra]

        # Plant B (fully visible occluder) has occlusion_fraction=0 and occlusion_side="none"
        # Flag it so the training code can skip invisible/bo mask losses for these instances
        occ_frac = ann.get("occlusion_fraction", None)
        occ_side = ann.get("occlusion_side", None)
        new_ann["is_occluder"] = 1 if (occ_frac == 0.0 and occ_side == "none") else 0

        new_annotations.append(new_ann)

    # Keep only categories that actually appear in annotations
    used_ids = {a["category_id"] for a in new_annotations}
    id2cat = {c["id"]: c for c in data["categories"]}
    categories = [id2cat[cid] for cid in sorted(used_ids) if cid in id2cat]

    print(f"  Converted {len(new_annotations)} annotations, skipped {skipped}")
    print(f"  Categories: {len(categories)} (IDs: {[c['id'] for c in categories]})")
    return data["images"], categories, new_annotations, data.get("info", {})


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
    if not os.path.exists(src_json):
        print(f"  SKIP {split}: {src_json} not found")
        return
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
                        help="Source dataset root (flowers_amodal_output dir)")
    parser.add_argument("--dst", default=os.path.expanduser("~/datasets/flower_amodal"),
                        help="Output dataset root (default: ~/datasets/flower_amodal)")
    parser.add_argument("--train-json", default=None,
                        help="Train annotation JSON (default: <src>/annotations/flowers_train.json)")
    parser.add_argument("--val-json", default=None,
                        help="Val annotation JSON (default: <src>/annotations/flowers_val.json)")
    parser.add_argument("--test-json", default=None,
                        help="Test annotation JSON (default: <src>/annotations/flowers_test.json)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking (recommended for WSL2 ext4 speed)")
    args = parser.parse_args()

    src_images = os.path.join(args.src, "images")
    train_json = args.train_json or os.path.join(args.src, "annotations", "flowers_train.json")
    val_json   = args.val_json   or os.path.join(args.src, "annotations", "flowers_val.json")
    test_json  = args.test_json  or os.path.join(args.src, "annotations", "flowers_test.json")

    dst = os.path.expanduser(args.dst)

    convert_split("train", train_json, src_images, dst, args.copy_images)
    convert_split("val",   val_json,   src_images, dst, args.copy_images)
    convert_split("test",  test_json,  src_images, dst, args.copy_images)

    print(f"\nDone. Dataset at: {dst}")
    print(f"Set FLOWER_DATA_ROOT={dst} when training.")


if __name__ == "__main__":
    main()
