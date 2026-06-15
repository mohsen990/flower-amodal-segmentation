#!/usr/bin/env python3
"""
Compute F1 scores for the flower amodal model.

F1 is computed at IoU=0.50 by finding the point on the precision-recall
curve (stored inside COCOeval) that maximises 2*P*R / (P+R), then
averaging across the 12 flower categories.

Usage:
    FLOWER_DATA_ROOT=~/datasets/flower_amodal \
    /home/mohsen/det2env/bin/python3 eval_f1.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --model-weights output/flower_aisformer/model_final.pth
"""

import copy, itertools, io, contextlib, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import register_flower_dataset  # noqa — must come before detectron2 imports

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from pycocotools.coco import COCO

from aistron.config import add_aistron_config
from aistron.data import AmodalDatasetMapper
from aistron.evaluation import AmodalInstanceEvaluator
from aistron.evaluation.amodal_evaluation import _evaluate_predictions_on_coco
from aistron.evaluation.amodal_cocoeval import AmodalCOCOeval as COCOeval
import aisformer
from aisformer import add_aisformer_config


# ── helpers ──────────────────────────────────────────────────────────────────

def best_f1_from_coco_eval(coco_eval, iou=0.50):
    """
    For each category, walk the P-R curve at the given IoU threshold and
    return the maximum F1.  Then average over categories.
    """
    iou_idx = int(round((iou - 0.50) / 0.05))          # 0 → IoU=0.50
    prec    = coco_eval.eval['precision']               # [T, R, K, A, M]
    prec_at = prec[iou_idx, :, :, 0, -1]               # [R=101, K=12]
    rec_thr = coco_eval.params.recThrs                  # 101 points, 0..1

    per_cat = []
    for k in range(prec_at.shape[1]):
        p = prec_at[:, k]
        valid = p >= 0
        if not valid.any():
            per_cat.append(0.0)
            continue
        p, r = p[valid], rec_thr[valid]
        denom = p + r
        f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
        per_cat.append(float(f1.max()))

    mean_f1 = float(np.mean(per_cat))
    return mean_f1, per_cat


# ── patched evaluator that exposes coco_eval objects ─────────────────────────

class F1Evaluator(AmodalInstanceEvaluator):
    """Same as AmodalInstanceEvaluator but saves each COCOeval object."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_evals = {}          # task → COCOeval

    def _eval_predictions(self, predictions, img_ids=None):
        import itertools, copy
        from detectron2.utils.file_io import PathManager
        import json

        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # remap category ids
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse = {v: k for k, v in
                       self._metadata.thing_dataset_id_to_contiguous_id.items()}
            num_cls = len(reverse)
            for r in coco_results:
                r["category_id"] = reverse[r["category_id"]]

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            fp = os.path.join(self._output_dir, "coco_instances_results.json")
            with PathManager.open(fp, "w") as f:
                f.write(json.dumps(coco_results))

        from collections import OrderedDict
        self._results = OrderedDict()
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task,
                    cocoeval_fn=COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if coco_results else None
            )
            self.coco_evals[task] = coco_eval          # ← save it
            res = self._derive_coco_results(
                coco_eval, task,
                class_names=self._metadata.get("thing_classes"),
            )
            self._results[task] = res


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",    required=True)
    parser.add_argument("--model-weights",  default=None,
                        help="Path to .pth (default: output_dir/model_final.pth)")
    parser.add_argument("--dataset",        default=None,
                        help="Dataset name (default: first TEST dataset in config)")
    args = parser.parse_args()

    cfg = get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.freeze()

    # build model
    from train_net import Trainer
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    dataset_name = args.dataset or cfg.DATASETS.TEST[0]
    data_loader  = Trainer.build_test_loader(cfg, dataset_name)

    out_dir  = os.path.join(cfg.OUTPUT_DIR, "inference_f1")
    evaluator = F1Evaluator(dataset_name, output_dir=out_dir)

    print(f"\nRunning inference on {dataset_name} …")
    inference_on_dataset(model, data_loader, evaluator)

    # ── print F1 results ──────────────────────────────────────────────────
    class_names = MetadataCatalog.get(dataset_name).thing_classes
    TASKS_OF_INTEREST = ["amodal_segm", "visible_segm", "occluded_segm", "bbox"]

    print("\n" + "=" * 62)
    print(f"  F1 SCORES  (IoU = 0.50,  dataset = {dataset_name})")
    print("=" * 62)

    for task in TASKS_OF_INTEREST:
        coco_eval = evaluator.coco_evals.get(task)
        if coco_eval is None or coco_eval.eval is None:
            print(f"  {task:<20}  —  no predictions")
            continue
        mean_f1, per_cat = best_f1_from_coco_eval(coco_eval, iou=0.50)
        ap50 = coco_eval.stats[1] * 100
        ar   = coco_eval.stats[8] * 100
        print(f"\n  {task}")
        print(f"    Mean F1  : {mean_f1*100:6.2f}%")
        print(f"    AP@50    : {ap50:6.2f}%")
        print(f"    AR@100   : {ar:6.2f}%")
        if class_names:
            print("    Per-category F1:")
            for name, f1 in zip(class_names, per_cat):
                print(f"      {name:<30}  {f1*100:5.2f}%")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
