import webdataset as wds
from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets
from src.datasets.cifar10.base import (
    get_train_transforms,
    get_eval_transforms,
    get_size,
)
import torch

from src.libraries.webdataset import build_loader

DATASET = CIFAR10Datasets["webdataset"]


class WebdatasetLoader(DataLoader):
    def _get(self, mode, transform, **kwargs):

        if self.remote:
            dataset = DATASET.get_remote(
                mode=mode,
                transforms=transform,
                filtering=self.filtering,
                filtering_classes=self.filtering_classes,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )
        else:
            dataset = DATASET.get_local(
                mode=mode,
                transforms=transform,
                filtering=self.filtering,
                filtering_classes=self.filtering_classes,
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
        t = get_train_transforms()
        mode = "train"
        return self._get(mode, t, **kwargs)

    def get_val_loader(self, **kwargs):
        t = get_eval_transforms()
        mode = "val"
        return self._get(mode, t, **kwargs)

    def get_test_loader(self, **kwargs):
        t = get_eval_transforms()
        mode = "test"
        return self._get(mode, t, **kwargs)
