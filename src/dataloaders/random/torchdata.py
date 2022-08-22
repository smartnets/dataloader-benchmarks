from src.dataloaders.base import DataLoader
import torch
from src.datasets.random.base import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
)
from src.datasets.random.index import RandomDatasets
from src.libraries.torchdata import filter_by_class

DATASET = RandomDatasets["torchdata"]


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

    # def _filter(self, dataset, filtering):
    #     """
    #     Filtering by labels in torchdata
    #     Based on mapping of label->integer for cifar10
    #     """
    #     print(f"Filtering for class names : {filtering}")
    #     # dp = dataset.to_iter_datapipe()
    #     dp = dataset
    #     # labels = [‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’]
    #     # so dp[i][1] == 0 means label == "airplane"
    #     labels_list = [LABELS_DICT[i] for i in filtering]
    #     dp = dp.filter(filter_fn=lambda x: x[1] in labels_list)

    #     return dp
