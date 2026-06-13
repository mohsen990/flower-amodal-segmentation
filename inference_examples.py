"""
Run AISFormer inference on flower images and save side-by-side visualisations.

  Left  — predicted amodal masks overlaid on the image
  Right — GT: green=visible, red=hidden

Usage:
    cd ~/flower_aistron
    FLOWER_DATA_ROOT=/mnt/d/flower_aistron \
    python inference_examples.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --weights ./output/flower_aisformer/model_best.pth \
        --split val --num-images 20 \
        --output-dir /mnt/d/flower_inference
"""

import argparse
import os
import random
import cv2
import numpy as np
import pycocotools.mask as mask_util

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultPredictor
from aistron.config import add_aistron_config
import aisformer
from aisformer import add_aisformer_config
import register_flower_dataset  # noqa: F401


def decode_rle(segm, h, w):
    if isinstance(segm, dict):
        return mask_util.decode(segm).astype(bool)
    return np.zeros((h, w), dtype=bool)


def overlay_pred_masks(img_bgr, instances):
    out = img_bgr.copy().astype(np.float32)
    field = ("pred_amodal_masks" if instances.has("pred_amodal_masks")
             else "pred_masks" if instances.has("pred_masks") else None)
    if field is None:
        return img_bgr
    masks = instances.get(field).numpy()
    for mask in masks:
        color = np.random.randint(50, 220, 3).tolist()
        colored = np.zeros_like(out)
        colored[mask > 0.5] = color
        out = cv2.addWeighted(out, 1.0, colored, 0.45, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def gt_overlay(img_bgr, annotations, h, w, alpha=0.45):
    overlay = img_bgr.copy()
    for ann in annotations:
        amodal  = decode_rle(ann.get("amodal_segm"),  h, w)
        visible = decode_rle(ann.get("visible_segm"), h, w)
        hidden  = amodal & ~visible
        overlay[visible] = (0, 200, 0)
        overlay[hidden]  = (0, 0, 220)
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def add_label(img_bgr, text, font_scale=0.8):
    out = np.ascontiguousarray(img_bgr.copy())
    cv2.putText(out, text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        default="configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default="/mnt/d/flower_inference")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.freeze()

    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(f"flower_amodal_{args.split}")
    eligible      = [d for d in dataset_dicts if len(d.get("annotations", [])) >= 1]
    sample        = random.sample(eligible, min(args.num_images, len(eligible)))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running inference on {len(sample)} images → {args.output_dir}\n")

    for i, d in enumerate(sample):
        fname   = os.path.basename(d["file_name"])
        img_bgr = cv2.imread(d["file_name"])
        if img_bgr is None:
            print(f"  [{i+1}] Could not read {fname}")
            continue

        h, w      = img_bgr.shape[:2]
        instances = predictor(img_bgr)["instances"].to("cpu")

        p_left  = add_label(overlay_pred_masks(img_bgr, instances), "Predicted amodal masks")
        p_right = add_label(gt_overlay(img_bgr, d.get("annotations", []), h, w),
                            "GT: green=visible / red=hidden")

        gap  = np.full((h, 6, 3), 200, dtype=np.uint8)
        grid = np.concatenate([p_left, gap, p_right], axis=1)

        cv2.imwrite(os.path.join(args.output_dir, f"{i+1:02d}_{fname}"),
                    np.ascontiguousarray(grid))
        print(f"  [{i+1:02d}/{len(sample)}] {fname}  ({len(instances)} detections)")

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
