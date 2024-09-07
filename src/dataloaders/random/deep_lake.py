from src.dataloaders.base import DataLoader
from src.datasets.random.index import RandomDatasets
from src.datasets.random.base import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
)
from src.libraries.deep_lake import filter_by_class

DATASET = RandomDatasets["deep_lake"]


class DeepLakeLoader(DataLoader):
    transform = None

    def _get(self, mode, transform, **kwargs):

        # kwargs["num_workers"] = 0 # increased performance
        self.transform = transform

        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=None)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=None)

        if self.filtering:
            filter_conditions = " or ".join([f"labels == '{LABELS_DICT[c]}'" for c in self.filtering_classes])
            dataset = dataset.filter(filter_conditions)

        loader = dataset.pytorch(
            transform=self.transform_hub,
            # distributed=self.distributed,
            **kwargs
        )
        return loader

    def get_train_loader(self, **kwargs):
        t = get_train_transforms(to_pil=True)
        return self._get("train", t, **kwargs)

    def get_val_loader(self, **kwargs):
        t = get_eval_transforms()
        return self._get("val", t, **kwargs)

    def get_test_loader(self, **kwargs):
        t = get_eval_transforms()
        return self._get("test", t, **kwargs)

    def transform_hub(self, sample):
        sample["images"] = self.transform(sample["images"])
        return sample
