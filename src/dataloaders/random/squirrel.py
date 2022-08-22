from src.dataloaders.base import DataLoader
from src.datasets.random.base import get_train_transforms, get_eval_transforms
from src.datasets.random.index import RandomDatasets
import torch


DATASET = RandomDatasets["squirrel"]


class SquirrelLoader(DataLoader):
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

        kwargs["batch_size"] = None

        loader = torch.utils.data.DataLoader(dataset, **kwargs)
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
