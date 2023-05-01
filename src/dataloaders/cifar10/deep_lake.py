from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets
from src.datasets.cifar10.base import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
)
from src.libraries.deep_lake import filter_by_class

DATASET = CIFAR10Datasets["deep_lake"]


class DeepLakeLoader(DataLoader):
    transform = None

    def _get(self, mode, transform, **kwargs):
        self.transform = transform
        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=None)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=None)

        if self.filtering:
            FC = [LABELS_DICT[c] for c in self.filtering_classes]
            dataset = filter_by_class(dataset, FC)

        loader = dataset.pytorch(
            transform={
                'images': self.transform, 'labels': None},
            distributed=self.distributed,
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
