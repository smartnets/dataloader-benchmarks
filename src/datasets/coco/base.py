# %%
import numpy as np
from pathlib import Path
import subprocess
import zipfile


import torch
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import settings as st
from src.utils.distributed import get_worker_info
from src.datasets.coco.labels import LABELS

# %%
COCO_SHARED_DIR = Path(st.local_data_dir) / "coco" / "shared"
COCO_SHARED_DIR.mkdir(parents=True, exist_ok=True)

LABEL_DICT = dict((x, i) for i, x in enumerate(LABELS))

def get_size(mode):
    if mode == "train":
        return 118287
    elif mode == "val":
        return 5000

# %%
PARAMS = {
    "val": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "zip_name": COCO_SHARED_DIR / "val2017.zip",
        "unzipped_name": COCO_SHARED_DIR / "val2017",
        "final_name": COCO_SHARED_DIR / "val",
        "annotation_file": COCO_SHARED_DIR / "annotations" / "instances_val2017.json",
    },
    "train": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "zip_name": COCO_SHARED_DIR / "train2017.zip",
        "unzipped_name": COCO_SHARED_DIR / "train2017",
        "final_name": COCO_SHARED_DIR / "train",
        "annotation_file": COCO_SHARED_DIR / "annotations" / "instances_train2017.json",
    },
    "annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "zip_name": COCO_SHARED_DIR / "annotations_trainval2017.zip",
        "unzipped_name": COCO_SHARED_DIR / "annotations",
        "final_name": COCO_SHARED_DIR / "annotations",
    }
}

# %%


def download_mode(mode):

    if not PARAMS[mode]["final_name"].is_dir():
        ARGS = [
            "aria2c",
            "-x",
            "10",
            "-j",
            "10",
            "-d",
            COCO_SHARED_DIR,
            PARAMS[mode]["url"]
        ]

        # %%
        pid = subprocess.Popen(ARGS)
        pid.wait()

        # %%
        with zipfile.ZipFile(PARAMS[mode]["zip_name"]) as zip_ref:
            zip_ref.extractall(COCO_SHARED_DIR)

        # %%
        PARAMS[mode]["unzipped_name"].rename(PARAMS[mode]["final_name"])

        # %%
        PARAMS[mode]["zip_name"].unlink(missing_ok=True)  # remove file
    else:
        print(f"{mode} has already been downloaded, skipping.")


def _train_tform():

    tform_train = A.Compose([
        A.RandomSizedBBoxSafeCrop(width=64, height=64, erosion_rate=0.2),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # transpose_mask = True
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels', 'bbox_ids'], min_area=16, min_visibility=0.6))  # "" are all the fields that will be cut when a bounding box is cut.
    return tform_train


def _eval_tform():

    tform_train = A.Compose([
        A.Resize(64, 64),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),  # transpose_mask = True
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels', 'bbox_ids'], min_area=16, min_visibility=0.6))  # "" are all the fields that will be cut when a bounding box is cut.
    return tform_train


def core_transform(mode, preprocess, image, targets):

    image, targets = preprocess(image, targets)

    boxes = np.array(targets["boxes"])
    categories = np.array(targets["categories"])

    # filter empty boxes
    if boxes.shape[0] > 0:
        w, h = boxes[:, 2], boxes[:, 3]
        n_idx = np.where((w > 0) & (h > 0))[0]
        boxes, categories = boxes[n_idx], categories[n_idx]

    if mode == "train":
        transform = _train_tform()
    else:
        transform = _eval_tform()

    B = len(boxes)
    transformed = transform(
        image=image,
        bboxes=boxes,
        bbox_ids=np.arange(B),
        class_labels=categories
    )
    new_boxes = transformed["bboxes"]

    B_new = len(new_boxes)
    new_image = transformed["image"]
    if new_image.shape[0] == 3:
        h, w = new_image.shape[1:]
    elif new_image.shape[2] == 3:
        h, w = new_image.shape[:2]

    labels_torch = torch.tensor(
        np.round(transformed['class_labels']), dtype=torch.int64)

    boxes_torch = torch.zeros((B_new, 4), dtype=torch.int64)
    for b, box in enumerate(new_boxes):
        box = torch.tensor(box)
        box[2:] += box[:2]
        box = box.round()
        box[0::2].clamp_(min=0, max=w - 1)  # TODO
        box[1::2].clamp_(min=0, max=h - 1)
        if np.allclose(box[0], box[2]):
            box[0] -= 1
        if np.allclose(box[1], box[3]):
            box[1] -= 1

        boxes_torch[b, :] = box

    # Put annotations in a separate object
    target = {'labels': labels_torch, 'boxes': boxes_torch}
    return new_image, target


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, mode, transforms=None):
        img_folder=PARAMS[mode]["final_name"]
        ann_file=PARAMS[mode]["annotation_file"]
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.mode = mode

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        img = np.array(img)
        target = {
            "idx": idx,
            "boxes": [e["bbox"] for e in target],
            "categories": [e["category_id"] for e in target]
        }

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __iter__(self):
        total_workers, global_worker_id = get_worker_info()
        for i in range(get_size(self.mode)):
            if i % total_workers == global_worker_id:
                yield self.__getitem__(i)


def get_coco(mode, transforms):

    dataset = CocoDetection(
        mode,
        transforms=transforms
    )
    return dataset
