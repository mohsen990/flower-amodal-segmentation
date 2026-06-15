"""Dataset registration for the Flower Amodal Dataset.

Expected layout (produced by convert_flower_dataset.py):
  <FLOWER_DATA_ROOT>/
    annotations/
      train_amodal.json
      val_amodal.json
      test_amodal.json
    train/images/
    val/images/
    test/images/

Override the root via env var:
  FLOWER_DATA_ROOT=/path/to/flower_amodal python train_net.py ...
"""

import os
from aistron.data.datasets.coco_amodal import register_aistron_cocolike_instances

DATASET_ROOT = os.environ.get(
    "FLOWER_DATA_ROOT",
    os.path.expanduser("~/datasets/flower_amodal"),
)

_SPLITS = {
    "flower_amodal_train": ("train_amodal.json", "train/images"),
    "flower_amodal_val":   ("val_amodal.json",   "val/images"),
    "flower_amodal_test":  ("test_amodal.json",  "test/images"),
}

for dataset_name, (ann_file, img_subdir) in _SPLITS.items():
    json_path = os.path.join(DATASET_ROOT, "annotations", ann_file)
    img_path  = os.path.join(DATASET_ROOT, img_subdir)
    if os.path.exists(json_path):
        register_aistron_cocolike_instances(dataset_name, {}, json_path, img_path)
        print(f"Registered {dataset_name} from {json_path}")
    else:
        print(f"Skipped {dataset_name}: {json_path} not found")
