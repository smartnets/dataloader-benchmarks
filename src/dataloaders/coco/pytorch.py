from src.dataloaders.base import DataLoader
from torchvision import transforms

import torch.utils.data as torch_data
from src.datasets.coco.base import core_transform, LABEL_DICT
from src.datasets.coco.index import CocoDatasets
from src.libraries.pytorch import filter_by_multi_class
from functools import partial

DATASET = CocoDatasets["pytorch"]


def collate_fn(batch):
    return tuple(zip(*batch))


def identity(x, y):
    return x, y


class PytorchLoader(DataLoader):
    def _get(self, mode, transform, **kwargs):

        t = partial(core_transform, mode, identity)
        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=t)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=t)

        sampler = None
        if self.filtering:
            FC = [LABEL_DICT[k] for k in self.filtering_classes]
            sampler = filter_by_multi_class(FC, dataset)

        if self.distributed:
            sampler = self.get_distributd_sampler(dataset)

        loader = torch_data.DataLoader(
            dataset, collate_fn=collate_fn, **kwargs, sampler=sampler
        )
        return loader

    def get_train_loader(self, **kwargs):
        return self._get("train", None, **kwargs)

    def get_val_loader(self, **kwargs):
        return self._get("val", None, **kwargs)
