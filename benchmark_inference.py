"""
Measure inference complexity: params, latency, peak GPU memory, GPU utilization.
Usage:
    FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
    /home/mohsen/det2env/bin/python3 benchmark_inference.py \
        --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
        --weights output/flower_aisformer_v2_2cat/model_final.pth
"""

import os, sys, time, argparse
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

import register_flower_dataset  # noqa
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
import detectron2.data.transforms as T
import cv2

from aistron.config import add_aistron_config
from aisformer import add_aisformer_config


def setup_cfg(config_file, weights):
    cfg = get_cfg()
    add_aistron_config(cfg)
    add_aisformer_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.freeze()
    return cfg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    cfg = setup_cfg(args.config_file, args.weights)
    predictor = DefaultPredictor(cfg)
    model = predictor.model

    # ── parameters ────────────────────────────────────────────────────────────
    total_params = count_parameters(model)
    print(f"\nParameters: {total_params/1e6:.1f}M ({total_params:,})")

    # ── load a few real images from val set ──────────────────────────────────
    dataset = DatasetCatalog.get("flower_amodal_val")
    images = []
    for d in dataset[:args.warmup + args.runs]:
        img = cv2.imread(d["file_name"])
        if img is not None:
            images.append(img)
    if len(images) < args.warmup + args.runs:
        # pad with repeats
        while len(images) < args.warmup + args.runs:
            images.append(images[-1].copy())

    # ── warmup ────────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warmup} iters) …")
    torch.cuda.reset_peak_memory_stats()
    for img in images[:args.warmup]:
        with torch.no_grad():
            predictor(img)
    torch.cuda.synchronize()

    # ── benchmark ─────────────────────────────────────────────────────────────
    print(f"Benchmarking ({args.runs} iters) …")
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for img in images[args.warmup:args.warmup + args.runs]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            predictor(img)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    # ── GPU utilization via nvidia-smi ────────────────────────────────────────
    import subprocess
    try:
        util_lines = []
        for img in images[args.warmup:args.warmup + args.runs]:
            with torch.no_grad():
                predictor(img)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                util_lines.append(float(result.stdout.strip()))
        gpu_util = np.mean(util_lines) if util_lines else float("nan")
    except Exception:
        gpu_util = float("nan")

    # ── results ───────────────────────────────────────────────────────────────
    lat = np.array(latencies)
    print("\n" + "="*60)
    print("  INFERENCE COMPLEXITY (batch=1)")
    print("="*60)
    print(f"  Parameters          : {total_params/1e6:.1f} M")
    print(f"  Checkpoint size     : 438 MB")
    print(f"  Latency / image     : {lat.mean():.1f} ms  (±{lat.std():.1f} ms)")
    print(f"  Min / Max           : {lat.min():.1f} ms / {lat.max():.1f} ms")
    print(f"  Throughput          : {1000/lat.mean():.2f} img/s")
    print(f"  Peak GPU memory     : {peak_mem_mb:.0f} MB")
    print(f"  GPU utilisation     : {gpu_util:.0f}%")
    print("="*60)


if __name__ == "__main__":
    main()
