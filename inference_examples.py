"""
Run inference and save per-instance visualisations.
  One row per flower instance, sorted front → back (by occlusion fraction).
  Each row = 3 panels:
    [Original] [Pred: PRED_COLORS fill + amodal contour] [GT: green=visible / red=hidden]

Usage:
    cd ~/flower_aistron
    FLOWER_DATA_ROOT=~/datasets/flower_amodal \
    /home/mohsen/det2env/bin/python3 inference_examples.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --weights output/flower_aisformer/model_final.pth \
        --split val --num-images 25 --output-dir /mnt/e/flower_inference_examples
"""

import argparse
import os
import sys
import random
import cv2
import numpy as np
import pycocotools.mask as mask_util

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

import register_flower_dataset  # noqa: F401

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from aistron.config import add_aistron_config
from aisformer import add_aisformer_config


# ── per-instance colours (RGB) ────────────────────────────────────────────────
PRED_COLORS = [
    (138,  43, 226),   # blue-violet
    (255, 140,   0),   # dark orange
    (220,  20,  60),   # crimson
    (  0, 180, 180),   # teal
    ( 50, 205,  50),   # lime green
    (255, 215,   0),   # gold
]


def setup_cfg(config_file, weights, score_thresh=0.5):
    cfg = get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


def decode_rle(segm, h, w):
    if isinstance(segm, dict):
        return mask_util.decode(segm).astype(bool)
    return np.zeros((h, w), dtype=bool)


def add_label(img_rgb, text, font_scale=0.85):
    out = np.ascontiguousarray(img_rgb.copy())
    bgr = out[:, :, ::-1].copy()
    cv2.putText(bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr[:, :, ::-1].copy()


# ── occlusion helpers ─────────────────────────────────────────────────────────

def ann_occlusion_fraction(ann, h, w):
    frac = ann.get("occlusion_fraction")
    if frac is not None:
        return float(frac)
    amodal  = decode_rle(ann.get("amodal_segm"),  h, w)
    visible = decode_rle(ann.get("visible_segm"), h, w)
    total = amodal.sum()
    if total == 0:
        return 0.0
    return float((amodal & ~visible).sum()) / float(total)


def pred_occlusion_fraction(instances, idx):
    if not instances.has("pred_amodal_masks"):
        return 0.0
    amodal = instances.pred_amodal_masks[idx].numpy()
    if amodal.ndim == 3:
        amodal = amodal[0]
    amodal = amodal.astype(bool)
    total = amodal.sum()
    if total == 0:
        return 0.0
    if instances.has("pred_visible_masks"):
        visible = instances.pred_visible_masks[idx].numpy()
        if visible.ndim == 3:
            visible = visible[0]
        visible = visible.astype(bool)
        return float((amodal & ~visible).sum()) / float(total)
    return 0.0


# ── single-instance panel drawers ─────────────────────────────────────────────

def draw_pred_instance(img_rgb, instances, idx, fill_alpha=0.45, contour_thick=2):
    """PRED_COLORS fill + precise amodal contour for ONE predicted instance (no bbox)."""
    result = img_rgb.copy()
    color  = PRED_COLORS[idx % len(PRED_COLORS)]

    if not instances.has("pred_amodal_masks"):
        return result

    mask = instances.pred_amodal_masks[idx].numpy()
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.astype(bool)

    overlay = result.copy()
    overlay[mask] = color
    result = cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0)

    # smooth the mask slightly before contouring so the outline hugs the
    # flower boundary without tracing every pixel-level noise artifact
    src = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    src_smooth = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    src_smooth = cv2.morphologyEx(src_smooth, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(src_smooth, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_TC89_KCOS)
    bgr = result[:, :, ::-1].copy()
    # black shadow for contrast, then colored line on top
    cv2.drawContours(bgr, contours, -1, (0, 0, 0), contour_thick + 2)
    cv2.drawContours(bgr, contours, -1, (color[2], color[1], color[0]), contour_thick)
    return bgr[:, :, ::-1]


def draw_gt_instance(img_bgr, ann, h, w):
    """GT for ONE annotation: dark green=visible, vivid blue=hidden."""
    amodal  = decode_rle(ann.get("amodal_segm"),  h, w)
    visible = decode_rle(ann.get("visible_segm"), h, w)
    hidden  = amodal & ~visible

    result = img_bgr.copy()

    # visible region — dark forest green, 70 % opaque
    ov_vis = result.copy()
    ov_vis[visible] = (20, 140, 20)          # BGR dark green
    result = cv2.addWeighted(ov_vis, 0.70, result, 0.30, 0)

    # hidden region — vivid royal blue, 85 % opaque (maximally distinct from green/yellow)
    ov_hid = result.copy()
    ov_hid[hidden] = (210, 30, 30)           # BGR deep blue
    result = cv2.addWeighted(ov_hid, 0.85, result, 0.15, 0)

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        default="configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml")
    parser.add_argument("--weights",      required=True)
    parser.add_argument("--output-dir",   default="/mnt/e/flower_inference_examples")
    parser.add_argument("--num-images",   type=int,   default=25)
    parser.add_argument("--score-thresh", type=float, default=0.90)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    random.seed(args.seed)

    cfg       = setup_cfg(args.config_file, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)
    print(f"GPU: {torch.cuda.is_available()}   Model: {args.weights}")

    dataset_dicts = DatasetCatalog.get(f"flower_amodal_{args.split}")
    eligible      = [d for d in dataset_dicts if len(d.get("annotations", [])) >= 1]
    sample        = random.sample(eligible, min(args.num_images, len(eligible)))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving {len(sample)} results → {args.output_dir}\n")

    for i, d in enumerate(sample):
        fname   = os.path.basename(d["file_name"])
        img_bgr = cv2.imread(d["file_name"])
        if img_bgr is None:
            print(f"  [{i+1}] Could not read {fname}")
            continue

        instances   = predictor(img_bgr)["instances"].to("cpu")
        img_rgb     = img_bgr[:, :, ::-1]
        annotations = d.get("annotations", [])
        h, w        = img_rgb.shape[:2]

        print(f"  [{i+1:02d}/{len(sample)}] {fname}: {len(instances)} detection(s)")

        # sort GT front → back (ascending occlusion fraction)
        ann_order = sorted(range(len(annotations)),
                           key=lambda k: ann_occlusion_fraction(annotations[k], h, w))

        # sort Pred front → back
        n_pred     = len(instances)
        pred_order = sorted(range(n_pred),
                            key=lambda k: pred_occlusion_fraction(instances, k))

        n_rows    = max(n_pred, len(ann_order), 1)
        gap_v     = np.ones((h, 6, 3), dtype=np.uint8) * 180

        rows = []
        for row_idx in range(n_rows):
            label = f"flower {row_idx + 1}"

            # Panel 1: original
            p1 = add_label(np.ascontiguousarray(img_rgb), "Original")

            # Panel 2: prediction for this instance
            if row_idx < n_pred:
                pi = pred_order[row_idx]
                p2 = add_label(draw_pred_instance(img_rgb, instances, pi),
                               f"Pred: {label}")
            else:
                p2 = add_label(np.ascontiguousarray(img_rgb),
                               f"Pred: {label} (no det.)")

            # Panel 3: GT for this annotation
            if row_idx < len(ann_order):
                ai = ann_order[row_idx]
                gt_rgb = draw_gt_instance(img_bgr, annotations[ai], h, w)[:, :, ::-1]  # BGR→RGB
                p3 = add_label(gt_rgb, f"GT: {label}")
            else:
                p3 = add_label(np.ascontiguousarray(img_rgb),
                               f"GT: {label} (no ann.)")

            rows.append(np.concatenate([p1, gap_v, p2, gap_v, p3], axis=1))

        # stack rows with horizontal dividers
        gap_h = np.ones((6, w * 3 + 6 * 2, 3), dtype=np.uint8) * 180
        grid_parts = []
        for r_idx, row in enumerate(rows):
            grid_parts.append(row)
            if r_idx < len(rows) - 1:
                grid_parts.append(gap_h)
        grid = np.concatenate(grid_parts, axis=0)

        out_path = os.path.join(args.output_dir, f"{i+1:02d}_{fname}")
        cv2.imwrite(out_path, np.ascontiguousarray(grid[:, :, ::-1]))

    print(f"\nDone. Open {args.output_dir} in Windows Explorer.")


if __name__ == "__main__":
    main()
