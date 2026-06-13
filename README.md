# flower_aistron

Amodal instance segmentation of flowers using [AISFormer](https://github.com/UARK-AICV/AISFormer), built on the aistron framework.

## Setup

```bash
# 1. Install detectron2 (match your CUDA/torch version)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 2. Install this package
cd ~/flower_aistron
pip install -e .
```

## Dataset

Your source dataset must be COCO-like with two extra fields per annotation:
- `segmentation` — visible (modal) mask
- `amodal_segmentation` — full amodal mask (RLE)

Convert to aistron format:

```bash
python convert_flower_dataset.py \
    --src /mnt/d/FlowerDataset \
    --dst ~/datasets/flower_aistron \
    --copy-images          # omit to use symlinks
```

Expected output:
```
flower_aistron/
├── annotations/
│   ├── train_amodal.json
│   └── val_amodal.json
├── train/images/
└── val/images/
```

## Training

```bash
cd ~/flower_aistron
FLOWER_DATA_ROOT=~/datasets/flower_aistron \
python train_net.py \
    --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
    --num-gpus 1
```

Key config knobs (`configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml`):
- `MODEL.ROI_HEADS.NUM_CLASSES` — number of flower categories (default `1`)
- `SOLVER.MAX_ITER` / `SOLVER.STEPS` — training schedule
- `OUTPUT_DIR` — where checkpoints are saved

## Evaluation

```bash
FLOWER_DATA_ROOT=~/datasets/flower_aistron \
python train_net.py \
    --config-file configs/Flower/aisformer_R50_flower_56x56_imagenet.yaml \
    --eval-only \
    MODEL.WEIGHTS ./output/flower_aisformer/model_best.pth
```

## Inference visualisation

```bash
FLOWER_DATA_ROOT=~/datasets/flower_aistron \
python inference_examples.py \
    --weights ./output/flower_aisformer/model_best.pth \
    --split val --num-images 20 \
    --output-dir /mnt/d/flower_inference
```
