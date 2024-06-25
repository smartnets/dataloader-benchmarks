from src.dataloaders.base import DataLoader
from src.datasets.coco.index import CocoDatasets
from src.datasets.coco.base import LABEL_DICT, core_transform
from src.libraries.deep_lake import filter_by_class
from functools import partial

DATASET = CocoDatasets["deep_lake"]


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


class DeepLakeLoader(DataLoader):
    transform = None

    def _get(self, mode, **kwargs):

        # kwargs["num_workers"] = 0 # increased performance
        # self.transform = partial(aux, mode)

        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=None)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=None)

        if self.filtering:
            FC = [LABEL_DICT[c] for c in self.filtering_classes]
            dataset = filter_by_class(dataset, FC)

        loader = dataset.pytorch(
            transform=partial(aux, mode),
            collate_fn=collate_fn,
            **kwargs
        )
        return loader

    def get_train_loader(self, **kwargs):
        return self._get("train", **kwargs)

    def get_val_loader(self, **kwargs):
        return self._get("val", **kwargs)

    # def transform_hub(self, sample):
    #     sample["images"] = self.transform(sample["images"])
    #     return sample
