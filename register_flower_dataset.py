"""Dataset registration for Flower Amodal Dataset."""
import os
from aistron.data.datasets.coco_amodal import register_aistron_cocolike_instances

# Override with env var: FLOWER_DATA_ROOT=/mnt/d/flower_aistron python ...
DATASET_ROOT = os.environ.get("FLOWER_DATA_ROOT", "/mnt/d/flower_aistron")

train_json = os.path.join(DATASET_ROOT, "annotations", "train_amodal.json")
train_imgs = os.path.join(DATASET_ROOT, "train", "images")
if os.path.exists(train_json):
    register_aistron_cocolike_instances(
        "flower_amodal_train", {}, train_json, train_imgs,
    )

val_json = os.path.join(DATASET_ROOT, "annotations", "val_amodal.json")
val_imgs = os.path.join(DATASET_ROOT, "val", "images")
if os.path.exists(val_json):
    register_aistron_cocolike_instances(
        "flower_amodal_val", {}, val_json, val_imgs,
    )

test_json = os.path.join(DATASET_ROOT, "annotations", "test_amodal.json")
test_imgs = os.path.join(DATASET_ROOT, "test", "images")
if os.path.exists(test_json):
    register_aistron_cocolike_instances(
        "flower_amodal_test", {}, test_json, test_imgs,
    )

print(f"Registered Flower datasets from {DATASET_ROOT}")
