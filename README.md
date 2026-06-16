# flower_aistron

Amodal instance segmentation of flowers using [AISFormer](https://github.com/UARK-AICV/AISFormer),
built on the [aistron](https://github.com/UARK-AICV/aistron) framework (Detectron2-based).

---

## Results — FlowerAmodal v3

**Validation set** (1,516 images) — model_final.pth (30k iterations, N_LAYERS=2, FP16):

| Task | AP | AP50 | AP75 | APl | AR |
|---|---|---|---|---|---|
| amodal_segm | 85.12 | 98.35 | 95.46 | 88.37 | 69.41 |
| visible_segm | 90.11 | 98.44 | 97.23 | 92.03 | 71.84 |
| bbox | 81.83 | 98.40 | 95.34 | 81.83 | 85.90 |

**Test set** (1,586 images, unseen from training) — mask IoU:

| IoU threshold | F1 | Precision | Recall | Confidence |
|---|---|---|---|---|
| ≥ 0.50 | **0.9834** | 0.9936 | 0.9735 | 0.90 |
| ≥ 0.75 | **0.9548** | 0.9646 | 0.9451 | 0.90 |

**Inference** (RTX 3050 Ti, 4 GB VRAM, batch=1):

| Metric | Value |
|---|---|
| Parameters | 57.4 M |
| Checkpoint size | 438 MB |
| Latency / image | 146.2 ms ± 33.2 ms |
| Throughput | 6.84 img/s |
| Peak GPU memory | 966 MB (24% of 4 GB) |
| GPU utilisation | 67% |

---

## Setup

```bash
# Python environment (det2env has Detectron2 + all dependencies)
/home/mohsen/det2env/bin/python3 --version

# Clone and enter project
cd ~/flower_aistron
```

---

## Dataset — FlowerAmodal v3

**Source:** 2-category subset of Oxford 102 Flower dataset (buttercup, bishop of llandaff).
**Annotation:** COCO format extended with `amodal_segm`, `visible_segm`, `occluder_segm`,
`amodal_bbox`, `visible_bbox`, `occlusion_fraction`, `is_occluder`.

### Step 1 — Convert source dataset

```bash
python convert_flower_dataset.py \
    --src /mnt/e/Dataset/flowers_v3_trainval \
    --dst ~/datasets/flower_amodal_v3 \
    --copy-images
```

Add test split separately (if in a different folder):

```bash
python convert_flower_dataset.py \
    --src /mnt/e/Dataset/flowers_v3_test \
    --dst ~/datasets/flower_amodal_v3 \
    --test-json /mnt/e/Dataset/flowers_v3_test/annotations/flowers_test.json \
    --copy-images
```

### Step 2 — (Optional) Generate synthetic CopyPaste occlusions

```bash
python generate_synthetic_occlusions.py \
    --src ~/datasets/flower_amodal_v3 \
    --dst ~/datasets/flower_amodal_v3_aug \
    --multiplier 2 \
    --overlap-min 0.10 --overlap-max 0.80
```

Expected layout:
```
~/datasets/flower_amodal_v3/
├── annotations/
│   ├── train_amodal.json   (6,062 images, 12,124 annotations)
│   ├── val_amodal.json     (1,516 images,  3,032 annotations)
│   └── test_amodal.json    (1,586 images,  3,172 annotations)
├── train/images/
├── val/images/
└── test/images/
```

---

## Training

```bash
cd ~/flower_aistron
FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
/home/mohsen/det2env/bin/python3 train_net.py \
    --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
    --num-gpus 1
```

Key config (configs/Flower/aisformer_R50_flower_v2_2cat.yaml):

| Parameter | Value |
|---|---|
| NUM_CLASSES | 2 |
| POOLER_RESOLUTION | 28 (→ 56×56 masks) |
| N_LAYERS | 2 |
| N_HEADS | 2 |
| EMB_DIM | 256 |
| MAX_ITER | 30,000 |
| AMP (FP16) | Enabled |
| BATCH_SIZE_PER_IMAGE | 16 |

---

## Evaluation (COCO AP)

```bash
FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
/home/mohsen/det2env/bin/python3 train_net.py \
    --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
    --eval-only \
    MODEL.WEIGHTS output/flower_aisformer_v2_2cat/model_final.pth
```

---

## F1 / Precision / Recall

```bash
# Validation set
FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
/home/mohsen/det2env/bin/python3 evaluate_f1_flower.py \
    --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
    --weights output/flower_aisformer_v2_2cat/model_final.pth \
    --split val --use-mask-iou --iou-thresholds 0.5 0.75

# Test set (unseen)
FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
/home/mohsen/det2env/bin/python3 evaluate_f1_flower.py \
    --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
    --weights output/flower_aisformer_v2_2cat/model_final.pth \
    --split test --use-mask-iou --iou-thresholds 0.5 0.75
```

---

## Inference Visualisation (3-panel: Original | Predicted | GT)

```bash
FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
/home/mohsen/det2env/bin/python3 inference_examples.py \
    --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
    --weights output/flower_aisformer_v2_2cat/model_final.pth \
    --split test --num-images 25 --output-dir /mnt/e/flower_inference_test
```

Panels: **Left** = original image · **Centre** = predicted amodal mask (colour fill + contour) ·
**Right** = GT (green = visible region, red = hidden/occluded region).
Instances sorted front→back by occlusion fraction.

---

## Inference Benchmark

```bash
FLOWER_DATA_ROOT=~/datasets/flower_amodal_v3 \
/home/mohsen/det2env/bin/python3 benchmark_inference.py \
    --config-file configs/Flower/aisformer_R50_flower_v2_2cat.yaml \
    --weights output/flower_aisformer_v2_2cat/model_final.pth
```

---

## Paper

The companion paper is in [paper/Aflower.tex](paper/Aflower.tex).
Figures are generated by:

```bash
cd paper/figures
/home/mohsen/det2env/bin/python3 fig_coco_metrics.py
/home/mohsen/det2env/bin/python3 fig_f1_convergence.py
/home/mohsen/det2env/bin/python3 fig_pipeline.py
```

Compile with:
```bash
cd paper
pdflatex Aflower.tex
pdflatex Aflower.tex   # second pass for references
```

---

## File Layout

```
flower_aistron/
├── train_net.py                  — main training / eval entry point
├── register_flower_dataset.py    — Detectron2 dataset registration
├── convert_flower_dataset.py     — source → aistron JSON conversion
├── generate_synthetic_occlusions.py — CopyPaste offline augmentation
├── evaluate_f1_flower.py         — F1/Precision/Recall sweep
├── inference_examples.py         — 3-panel visualisation
├── benchmark_inference.py        — latency / memory profiling
├── aisformer/
│   └── mask_head.py              — transformer mask head (is_occluder weighting)
├── aistron/data/
│   └── dataset_mapper.py         — amodal dataset mapper (gt_is_occluder tensor)
├── configs/Flower/
│   └── aisformer_R50_flower_v2_2cat.yaml — active training config
└── paper/
    ├── Aflower.tex               — IEEE conference paper
    └── figures/
        ├── fig_coco_metrics.py / .pdf
        ├── fig_f1_convergence.py / .pdf
        └── fig_pipeline.py / .pdf
```

---

## References

This project builds directly on:

| Project | Repo | Paper |
|---|---|---|
| **AISFormer** — transformer amodal mask head | [UARK-AICV/AISFormer](https://github.com/UARK-AICV/AISFormer) | Tran et al., BMVC 2022 |
| **aistron** — amodal segmentation framework (Detectron2) | [UARK-AICV/aistron](https://github.com/UARK-AICV/aistron) | — |
| **Detectron2** — object detection platform | [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) | Wu et al., 2019 |
| **Oxford 102 Flower Dataset** — source images | [VGG / Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102) | Nilsback & Zisserman, ICVGIP 2008 |

```bibtex
@inproceedings{tran2022aisformer,
  title     = {AISFormer: Amodal Instance Segmentation with Transformer},
  author    = {Tran, Minh and Vo, Khoa and Yamazaki, Kashu and Fernandes, Arthur and Kidd, Michael and Le, Ngan},
  booktitle = {Proc. British Machine Vision Conf. (BMVC)},
  year      = {2022}
}

@misc{nilsback2008flower,
  author    = {Nilsback, Maria-Elena and Zisserman, Andrew},
  title     = {Automated Flower Classification over a Large Number of Classes},
  booktitle = {Proc. Indian Conf. Computer Vision, Graphics and Image Processing (ICVGIP)},
  pages     = {722--729},
  year      = {2008}
}
```
