#!/usr/bin/env python
"""
Training script for Flower Amodal Instance Segmentation with AISFormer.

Usage:
    cd ~/flower_aistron
    FLOWER_DATA_ROOT=/mnt/d/flower_aistron \
    python train_net.py \
        --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
        --num-gpus 1 \
        OUTPUT_DIR ./output/flower_aisformer
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import DatasetEvaluators, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from aistron.data import AmodalDatasetMapper
from aistron.evaluation import AmodalInstanceEvaluator
from aistron.config import add_aistron_config

import aisformer
from aisformer import add_aisformer_config
import register_flower_dataset  # noqa: F401 — registers flower datasets


def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "coco_amodal":
        return AmodalInstanceEvaluator(dataset_name, output_dir=output_folder)
    raise NotImplementedError(
        f"No evaluator for dataset '{dataset_name}' with type '{evaluator_type}'"
    )


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=AmodalDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=AmodalDatasetMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logging.getLogger("detectron2.trainer").info(
            "Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name,
                                output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})


def setup(args):
    cfg = get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
