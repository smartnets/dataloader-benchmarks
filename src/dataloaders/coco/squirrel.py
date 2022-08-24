from src.dataloaders.base import DataLoader
from src.datasets.coco.base import core_transform
from src.datasets.coco.index import CocoDatasets
import torch
import numpy as np
from functools import partial


DATASET = CocoDatasets["squirrel"]


def identity(x, y):
    return x, y


def aux(mode, sample):
    image = sample["jpg"]
    labels = sample["labels"]
    boxes = sample["boxes"]
    target = {"categories": labels, "boxes": boxes}
    return core_transform(mode, identity, image, target)


class SquirrelLoader(DataLoader):
    def _get(self, mode, **kwargs):

        t = partial(aux, mode)
        if self.remote:
            dataset = DATASET.get_remote(
                mode=mode,
                transforms=t,
                filtering=self.filtering,
                filtering_classes=self.filtering_classes,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )
        else:
            dataset = DATASET.get_local(
                mode=mode,
                transforms=t,
                filtering=self.filtering,
                filtering_classes=self.filtering_classes,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )

        kwargs["batch_size"] = None

        loader = torch.utils.data.DataLoader(dataset, **kwargs)
        return loader

    def get_train_loader(self, **kwargs):
        return self._get("train", **kwargs)

    def get_val_loader(self, **kwargs):
        return self._get("val", **kwargs)
