from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets
from torchvision import transforms
from src.datasets.cifar10.base import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
)

import torch.utils.data as torch_data
from src.libraries.pytorch import filter_by_class


DATASET = CIFAR10Datasets["pytorch"]


class PytorchLoader(DataLoader):
    def _get(self, mode, transform, **kwargs):
        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=transform)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=transform)

        sampler = None
        if self.filtering:
            FC = [LABELS_DICT[k] for k in self.filtering_classes]
            sampler = filter_by_class(FC, dataset)

        if self.distributed:
            sampler = self.get_distributd_sampler(dataset)

        loader = torch_data.DataLoader(
            dataset,
            **kwargs,
            sampler=sampler,
        )
        return loader

    def get_train_loader(self, **kwargs):
        t = get_train_transforms()
        return self._get("train", t, **kwargs)

    def get_val_loader(self, **kwargs):
        t = get_eval_transforms()
        return self._get("val", t, **kwargs)

    def get_test_loader(self, **kwargs):
        t = get_eval_transforms()
        return self._get("test", t, **kwargs)
