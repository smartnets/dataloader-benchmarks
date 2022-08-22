from src.dataloaders.base import DataLoader
from src.datasets.cifar10.index import CIFAR10Datasets

import torch
import torchvision
from typing import List

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
import ffcv.loader
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from src.datasets.cifar10.base import get_cifar10

DATASET = CIFAR10Datasets["ffcv"]

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]


def get_transforms(mode, rank):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice(f"cuda:{rank}"),
        Squeeze(),
    ]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    image_pipeline.extend(
        [
            ToTensor(),
            ToDevice(f"cuda:{rank}", non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return image_pipeline, label_pipeline


class FFCVLoader(DataLoader):
    def __init__(
        self,
        filtering: bool = False,
        filtering_classes: list = None,
        remote: bool = False,
        distributed: bool = False,
        world_size: int = None,
        rank: int = None,
    ):
        super().__init__(
            filtering, filtering_classes, True, remote, distributed, world_size, rank
        )

    def get_train_loader(self, **kwargs):
        super().get_train_loader()

        mode = "train"
        image_pipeline, label_pipeline = get_transforms(mode, self.rank)

        # Create loaders
        loader = ffcv.loader.Loader(
            DATASET.get_local_path() / f"{mode}/cifar10-{mode}.beton",
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            order=ffcv.loader.OrderOption.RANDOM,
            distributed=(mode == "train") and (self.distributed),
            drop_last=(mode == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
        return loader

    def get_val_loader(self, **kwargs):
        mode = "val"
        image_pipeline, label_pipeline = get_transforms(mode, self.rank)

        # Create loaders
        loader = ffcv.loader.Loader(
            DATASET.get_local_path() / f"{mode}/cifar10-{mode}.beton",
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            order=ffcv.loader.OrderOption.RANDOM,
            drop_last=(mode == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
        return loader

    def get_test_loader(self, **kwargs):
        mode = "test"
        image_pipeline, label_pipeline = get_transforms(mode, self.rank)

        loader = ffcv.loader.Loader(
            DATASET.get_local_path() / f"{mode}/cifar10-{mode}.beton",
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            order=ffcv.loader.OrderOption.RANDOM,
            drop_last=(mode == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
        return loader
