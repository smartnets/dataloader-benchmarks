from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets
import torch
from src.datasets.cifar10.base import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
)
from src.libraries.torchdata import filter_by_class

DATASET = CIFAR10Datasets["torchdata"]


class TorchdataLoader(DataLoader):
    def _get(self, mode, transform, **kwargs):

        if self.remote:
            dataset = DATASET.get_remote(
                mode=mode,
                transforms=transform,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )
        else:
            dataset = DATASET.get_local(
                mode=mode,
                transforms=transform,
                distributed=self.distributed,
                batch_size=kwargs["batch_size"],
            )

        if self.filtering:
            dataset = filter_by_class(
                [LABELS_DICT[i] for i in self.filtering_classes], dataset
            )
        loader = torch.utils.data.DataLoader(dataset, **kwargs)
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
