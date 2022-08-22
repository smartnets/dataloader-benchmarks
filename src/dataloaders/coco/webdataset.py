import torch.utils.data as torch_data
from src.dataloaders.base import DataLoader
from src.datasets.coco.index import CocoDatasets
from torchvision import transforms

import numpy as np
import webdataset as wds
from src.datasets.coco.base import core_transform, get_size
from functools import partial
import torch
from src.libraries.webdataset import build_loader


DATASET = CocoDatasets["webdataset"]


def identity(x, y):
    return np.array(x), y


def aux(mode, sample):
    image, labels, boxes = sample
    target = {"categories": labels, "boxes": boxes}
    return core_transform(mode, identity, image, target)


def collate_fn(batch):
    return tuple(zip(*batch))


class WebdatasetLoader(DataLoader):
    def _get(self, mode, **kwargs):
        t = partial(aux, mode)
        if self.remote:
            dataset = DATASET.get_remote(
                mode=mode,
                transforms=t,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )
        else:
            dataset = DATASET.get_local(
                mode=mode,
                transforms=t,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )

        loader = build_loader(
            dataset,
            get_size(mode),
            kwargs["batch_size"],
            kwargs["num_workers"],
            self.distributed,
        )

        return loader

    def get_train_loader(self, **kwargs):
        return self._get("train", **kwargs)

    def get_val_loader(self, **kwargs):
        return self._get("val", **kwargs)
