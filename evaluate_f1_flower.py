"""
Evaluate AISFormer on Flower dataset using F1/Precision/Recall metrics.

Usage:
    cd ~/flower_aistron

    # Val set (bbox IoU):
    FLOWER_DATA_ROOT=~/datasets/flower_amodal \
    /home/mohsen/det2env/bin/python3 evaluate_f1_flower.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --weights output/flower_aisformer/model_final.pth \
        --split val \
        --iou-thresholds 0.5 0.75

    # Test set:
    FLOWER_DATA_ROOT=~/datasets/flower_amodal \
    /home/mohsen/det2env/bin/python3 evaluate_f1_flower.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --weights output/flower_aisformer/model_final.pth \
        --split test \
        --iou-thresholds 0.5

    # Mask IoU (slower but more accurate):
    FLOWER_DATA_ROOT=~/datasets/flower_amodal \
    /home/mohsen/det2env/bin/python3 evaluate_f1_flower.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --weights output/flower_aisformer/model_final.pth \
        --split val --use-mask-iou
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
import cv2
import torch
from collections import defaultdict

# project root
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

import register_flower_dataset  # noqa: F401

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
import pycocotools.mask as mask_util

from aistron.config import add_aistron_config
from aisformer import add_aisformer_config


# ── config ────────────────────────────────────────────────────────────────────

def setup_cfg(config_file, weights, score_thresh=0.05):
    cfg = get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.freeze()
    return cfg


# ── IoU helpers ───────────────────────────────────────────────────────────────

def compute_iou_mask(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union        = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def compute_iou_bbox(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def decode_gt_segm(segm, h, w):
    if isinstance(segm, list):
        rles = mask_util.frPyObjects(segm, h, w)
        return mask_util.decode(mask_util.merge(rles)).astype(bool)
    elif isinstance(segm, dict):
        return mask_util.decode(segm).astype(bool)
    return np.zeros((h, w), dtype=bool)


# ── F1 evaluation ─────────────────────────────────────────────────────────────

def evaluate_predictions(per_image_data, iou_threshold=0.5, conf_thresholds=None):
    """
    per_image_data: list of dicts with keys:
        n_gt             – number of GT instances
        pairs            – list of (pred_idx, gt_idx, iou, score)
        all_pred_scores  – dict {pred_idx: score}  (preds with no GT pair)
    Returns: dict {conf_thr: {precision, recall, f1, tp, fp, fn, n_predictions}}
    """
    if conf_thresholds is None:
        conf_thresholds = np.arange(0.0, 1.0, 0.05).tolist()

    results = {}
    for conf_thr in conf_thresholds:
        total_tp = total_fp = total_fn = 0

        for img_data in per_image_data:
            n_gt  = img_data["n_gt"]
            pairs = img_data["pairs"]

            paired_preds = sorted(
                set(pi for pi, gi, iou, sc in pairs if sc >= conf_thr),
                key=lambda pi: max(sc for p, g, i, sc in pairs if p == pi),
                reverse=True,
            )
            unpaired_preds = [
                pi for pi, sc in img_data.get("all_pred_scores", {}).items()
                if sc >= conf_thr and pi not in set(paired_preds)
            ]
            active_preds = paired_preds + unpaired_preds

            pairs_by_pred = defaultdict(list)
            for pi, gi, iou, sc in pairs:
                pairs_by_pred[pi].append((gi, iou))

            gt_matched = [False] * n_gt
            tp = fp = 0
            for pi in active_preds:
                best_iou, best_gi = 0, -1
                for gi, iou in pairs_by_pred.get(pi, []):
                    if not gt_matched[gi] and iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_threshold and best_gi >= 0:
                    tp += 1
                    gt_matched[best_gi] = True
                else:
                    fp += 1
            fn = sum(1 for m in gt_matched if not m)
            total_tp += tp;  total_fp += fp;  total_fn += fn

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec)   if (prec + rec) > 0         else 0.0
        results[round(conf_thr, 2)] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "n_predictions": total_tp + total_fp,
        }
    return results


# ── inference helper ──────────────────────────────────────────────────────────

def run_one(predictor, bgr_img, use_mask_iou):
    out   = predictor(bgr_img)
    inst  = out["instances"]
    if len(inst) == 0:
        return []
    boxes  = inst.pred_boxes.tensor.cpu().numpy() if inst.has("pred_boxes") else []
    scores = inst.scores.cpu().numpy()            if inst.has("scores")     else []
    has_am = inst.has("pred_amodal_masks")
    results = []
    for j in range(len(scores)):
        m = None
        if has_am and use_mask_iou:
            m = inst.pred_amodal_masks[j].cpu().numpy().astype(np.float32)
            if m.ndim == 3:
                m = m[0]
        results.append((float(scores[j]), boxes[j].tolist(), m))
    return results


def merge_tta_masks(base_results, aug_list, H, W):
    merged = []
    for idx, (score, bbox, base_mask) in enumerate(base_results):
        if base_mask is None:
            merged.append({"score": score, "bbox": bbox, "mask": None})
            continue
        acc = base_mask.astype(np.float32).copy();  n = 1
        for aug_results, tfn in aug_list:
            if idx < len(aug_results):
                _, _, aug_mask = aug_results[idx]
                if aug_mask is not None:
                    acc += tfn(aug_mask, H, W);  n += 1
        merged.append({"score": score, "bbox": bbox, "mask": (acc / n) >= 0.5})
    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate F1/Precision/Recall on Flower amodal dataset")
    parser.add_argument("--config-file",      required=True)
    parser.add_argument("--weights",          required=True)
    parser.add_argument("--split",            default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--iou-thresholds",   nargs="+", type=float, default=[0.5])
    parser.add_argument("--score-thresh",     type=float, default=0.01)
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--use-mask-iou",     action="store_true",
                        help="Use amodal mask IoU instead of bbox IoU")
    parser.add_argument("--tta",              action="store_true",
                        help="Test-time augmentation: flip + 0.8×/1.2× scales")
    parser.add_argument("--data-root",
                        default=os.environ.get(
                            "FLOWER_DATA_ROOT",
                            os.path.expanduser("~/datasets/flower_amodal")))
    args = parser.parse_args()

    cfg       = setup_cfg(args.config_file, args.weights, args.score_thresh)
    predictor = DefaultPredictor(cfg)

    dataset_name  = f"flower_amodal_{args.split}"
    dataset_dicts = DatasetCatalog.get(dataset_name)

    print(f"\n{'='*70}")
    print(f"  F1 / Precision / Recall  —  Flower {args.split} set")
    print(f"{'='*70}")
    print(f"  Model   : {args.weights}")
    print(f"  Images  : {len(dataset_dicts)}")
    print(f"  IoU thr : {args.iou_thresholds}")
    print(f"  Mode    : {'mask IoU (amodal)' if args.use_mask_iou else 'bbox IoU'}")

    ann_file = os.path.join(args.data_root, "annotations", f"{args.split}_amodal.json")
    with open(ann_file) as f:
        gt_data = json.load(f)

    fname_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
    img_info    = {img["id"]: img              for img in gt_data["images"]}
    gt_by_image = defaultdict(list)
    for ann in gt_data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    print(f"  GT anns : {len(gt_data['annotations'])}")

    print(f"\n  Running inference …"
          + (" [TTA]" if args.tta else ""))

    per_image_data = []
    n_total_preds  = 0

    for i, d in enumerate(dataset_dicts):
        fname  = os.path.basename(d["file_name"])
        img_id = fname_to_id.get(fname)
        if img_id is None:
            continue

        img_meta = img_info[img_id]
        h, w     = img_meta["height"], img_meta["width"]
        gt_anns  = gt_by_image.get(img_id, [])

        img_bgr = cv2.imread(d["file_name"])
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]

        base_results = run_one(predictor, img_bgr, args.use_mask_iou)

        if args.tta and args.use_mask_iou and base_results:
            flip_bgr = np.ascontiguousarray(img_bgr[:, ::-1, :])
            flip_res = run_one(predictor, flip_bgr, True)
            def unflip(m, H, W):    return m[:, ::-1]

            s08_bgr = cv2.resize(img_bgr, (int(W * 0.8), int(H * 0.8)))
            s08_res = run_one(predictor, s08_bgr, True)
            def unscale08(m, H, W): return cv2.resize(m.astype(np.float32), (W, H),
                                                       interpolation=cv2.INTER_LINEAR)
            s12_bgr = cv2.resize(img_bgr, (int(W * 1.2), int(H * 1.2)))
            s12_res = run_one(predictor, s12_bgr, True)
            def unscale12(m, H, W): return cv2.resize(m.astype(np.float32), (W, H),
                                                       interpolation=cv2.INTER_LINEAR)
            preds = merge_tta_masks(base_results,
                                    [(flip_res, unflip),
                                     (s08_res, unscale08),
                                     (s12_res, unscale12)], H, W)
        else:
            preds = [{"score": sc, "bbox": bx,
                      "mask": (mk.astype(bool) if mk is not None else None)}
                     for sc, bx, mk in base_results]

        if args.use_mask_iou:
            gt_masks = [
                decode_gt_segm(a.get("amodal_segm", a.get("segmentation")), h, w)
                for a in gt_anns
            ]
        else:
            gt_masks = None

        pairs = []
        all_pred_scores = {}
        for pi, pred in enumerate(preds):
            all_pred_scores[pi] = pred["score"]
            for gi, gt_ann in enumerate(gt_anns):
                if args.use_mask_iou and pred["mask"] is not None \
                        and gt_masks is not None and gt_masks[gi] is not None:
                    iou = compute_iou_mask(pred["mask"], gt_masks[gi])
                else:
                    gt_bbox = gt_ann.get("amodal_bbox", gt_ann["bbox"])
                    gt_box  = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
                    iou = compute_iou_bbox(pred["bbox"], gt_box)
                pairs.append((pi, gi, iou, pred["score"]))

        per_image_data.append({
            "n_gt":            len(gt_anns),
            "pairs":           pairs,
            "all_pred_scores": all_pred_scores,
        })
        n_total_preds += len(preds)

        del preds, base_results, gt_masks
        del img_bgr
        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache();  gc.collect()
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(dataset_dicts)}  ({n_total_preds} preds so far)")

    print(f"  Done. Total predictions: {n_total_preds}")
    torch.cuda.empty_cache();  gc.collect()

    all_results = {}
    for iou_thr in args.iou_thresholds:
        print(f"\n{'='*70}")
        print(f"  Results @ IoU = {iou_thr}")
        print(f"{'='*70}")
        conf_thresholds = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]
        results = evaluate_predictions(
            per_image_data,
            iou_threshold=iou_thr,
            conf_thresholds=conf_thresholds,
        )

        print(f"\n  {'Conf':>6}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  "
              f"{'TP':>6}  {'FP':>6}  {'FN':>6}  {'#Pred':>6}")
        print(f"  {'-'*62}")

        best_f1, best_conf, best_row = 0.0, 0.0, None
        for conf, r in sorted(results.items()):
            print(f"  {conf:6.2f}  {r['precision']:8.4f}  {r['recall']:8.4f}  "
                  f"{r['f1']:8.4f}  {r['tp']:6d}  {r['fp']:6d}  {r['fn']:6d}  "
                  f"{r['n_predictions']:6d}")
            if r["f1"] > best_f1:
                best_f1, best_conf, best_row = r["f1"], conf, r

        print(f"\n  BEST F1 = {best_f1:.4f}  @ confidence = {best_conf:.2f}")
        if best_row:
            print(f"    Precision = {best_row['precision']:.4f}")
            print(f"    Recall    = {best_row['recall']:.4f}")
            print(f"    TP={best_row['tp']},  FP={best_row['fp']},  FN={best_row['fn']}")
        all_results[iou_thr] = results

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, f"f1_results_{args.split}.json")
        with open(out_file, "w") as f:
            json.dump({
                "config":  args.config_file,
                "weights": args.weights,
                "split":   args.split,
                "results": {str(k): {str(c): v for c, v in r.items()}
                            for k, r in all_results.items()},
            }, f, indent=2)
        print(f"\n  Results saved to: {out_file}")


if __name__ == "__main__":
    main()
