import typing as t
import numpy as np
from collections import defaultdict
from functools import partial
import torch
from squirrel.driver.msgpack import MessagepackDriver
from squirrel.iterstream.torch_composables import (
    SplitByRank,
    SplitByWorker,
    TorchIterable,
)

from torch.utils.data import default_collate

from src.datasets.transformations.custom import CustomTransform


def aug(augmentation, r):
    return augmentation(r)


def augmentation_multi_gpu(augmentation, r) -> t.Dict[str, t.Any]:
    return {"images": augmentation(r["jpg"]), "labels": r["cls"]}


def augmentation_single_gpu(augmentation, r):
    return (augmentation(r["jpg"]), r["cls"])


AUGMENTATION_MAPS = {
    "classification": {
        "single": augmentation_single_gpu,
        "multi": augmentation_multi_gpu,
    },
    "coco": {
        "single": aug,
        "multi": aug,
    },
}


def collate_coco(batch):
    return tuple(zip(*batch))


def filter_based_on_labels(active_labels, sample):
    return sample[1] in active_labels


COLLATE_FNS = {"classification": default_collate, "coco": collate_coco}


def build_dataset(
    path,
    transformations,
    distributed: bool = False,
    batch_size: int = 4,
    filtering: bool = False,
    filtering_classes: list = None,
    dataset_kind: str = "classification",
):

    # c_fn = collate if distributed else default_collate
    c_fn = COLLATE_FNS[dataset_kind]
    AM = AUGMENTATION_MAPS[dataset_kind]
    path = str(path)
    # t = CustomTransform(transformations, distributed)
    filter = partial(filter_based_on_labels, filtering_classes)

    if distributed:
        transformation_map = partial(AM["multi"], transformations)
    else:
        transformation_map = partial(AM["single"], transformations)

    hooks = [SplitByWorker]
    if distributed:
        hooks.append(SplitByRank)
    ds = MessagepackDriver(path).get_iter(key_hooks=hooks)

    if distributed:
        ds = ds.compose(SplitByRank)

    ds = ds.compose(SplitByWorker)
    ds = ds.map(transformation_map)

    if filtering:
        ds = ds.filter(filter)

    ds = ds.batched(batch_size, collation_fn=c_fn).compose(TorchIterable)
    return ds
