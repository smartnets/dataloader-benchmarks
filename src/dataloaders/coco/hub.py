import torch.utils.data as torch_data
from src.dataloaders.base import DataLoader
from src.datasets.coco.index import CocoDatasets
from torchvision import transforms

from src.datasets.coco.base import LABEL_DICT, core_transform
from functools import partial

from src.libraries.hub import filter_by_classs


DATASET = CocoDatasets["hub"]


def identity(x, y):
    t = {}
    t["categories"] = y["labels"]
    t["boxes"] = y["boxes"]
    return x, t


def aux(mode, sample):
    image = sample["images"]
    return core_transform(mode, identity, image, sample)


def collate_fn(batch):
    return tuple(zip(*batch))


class HubLoader(DataLoader):
    def _get(self, mode, **kwargs):
        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=None)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=None)

        if self.filtering:
            FC = [LABEL_DICT[x] for x in self.filtering_classes]
            dataset = filter_by_classs(dataset, FC)

        t = partial(aux, mode)
        loader = dataset.pytorch(transform=t, **kwargs, collate_fn=collate_fn)
        return loader

    def get_train_loader(self, **kwargs):
        return self._get("train", **kwargs)

    def get_val_loader(self, **kwargs):
        return self._get("val", **kwargs)
