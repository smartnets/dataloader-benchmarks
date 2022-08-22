from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets
from src.datasets.cifar10.base import get_train_transforms, get_eval_transforms
from torchvision import transforms

DATASET = CIFAR10Datasets["hub"]


class HubLoader(DataLoader):
    def _get(self, mode, transform, **kwargs):
        if self.remote:
            dataset = DATASET.get_remote(mode=mode, transforms=None)
        else:
            dataset = DATASET.get_local(mode=mode, transforms=None)

        if self.filtering:
            dataset = self._filter(dataset=dataset)

        # import pdb; pdb.set_trace()
        loader = dataset.pytorch(
            transform={"images": transform, "labels": None}, **kwargs
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

    def _filter(self, dataset):
        """
        Pythonic way of filtering in hub
        """
        class_query_str = f"labels == '{self.filtering_classes[0]}'"
        for i in range(1, len(self.filtering_classes)):
            class_query_str += f" or labels == '{self.filtering_classes[i]}'"
        return dataset.filter(class_query_str)
