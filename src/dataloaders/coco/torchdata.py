from src.dataloaders.base import DataLoader
import torch
from src.datasets.coco.base import core_transform, LABEL_DICT
from src.datasets.coco.index import CocoDatasets
from functools import partial
from src.libraries.torchdata import filter_by_multiclass

DATASET = CocoDatasets["torchdata"]


def identity(x, y):
    return (x, y)


def collate_fn(batch):
    return tuple(zip(*batch))


class TorchdataLoader(DataLoader):
    def _get(self, mode, transform, **kwargs):

        t = partial(core_transform, mode, identity)
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

        if self.filtering:
            dataset = filter_by_multiclass(
                [LABEL_DICT[i] for i in self.filtering_classes], dataset
            )

        loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **kwargs)
        return loader

    def get_train_loader(self, **kwargs):
        mode = "train"
        return self._get(mode, None, **kwargs)

    def get_val_loader(self, **kwargs):
        mode = "val"
        return self._get(mode, None, **kwargs)

    def _filter(self, dataset, filtering):
        pass
