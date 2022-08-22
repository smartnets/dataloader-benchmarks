from math import dist
from src.dataloaders.base import DataLoader

import torch
import torchvision
from typing import List

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
import ffcv.loader
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Cutout,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from src.datasets.random.index import RandomDatasets

DATASET = RandomDatasets["ffcv"]

CIFAR_MEAN = [50.0, 50.0, 50.0]
CIFAR_STD = [25.0, 25.0, 25.0]


def get_transforms(mode, rank):
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        ToDevice(f"cuda:{rank}", non_blocking=True),
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
    def get_train_loader(self, **kwargs):
        super().get_train_loader()

        mode = "train"
        image_pipeline, label_pipeline = get_transforms(mode, self.rank)

        # Create loaders
        loader = ffcv.loader.Loader(
            DATASET.get_local_path() / f"{mode}/random-{mode}.beton",
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            distributed=(self.distributed) and (mode == "train"),
            order=ffcv.loader.OrderOption.RANDOM,
            drop_last=(mode == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
        return loader

    def get_val_loader(self, **kwargs):
        mode = "val"
        image_pipeline, label_pipeline = get_transforms(mode, self.rank)

        # Create loaders
        loader = ffcv.loader.Loader(
            DATASET.get_local_path() / f"{mode}/random-{mode}.beton",
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            distributed=(mode == "train") and (self.distributed),
            order=ffcv.loader.OrderOption.RANDOM,
            drop_last=(mode == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
        return loader

    def get_test_loader(self, **kwargs):
        mode = "test"
        image_pipeline, label_pipeline = get_transforms(mode, self.rank)

        loader = ffcv.loader.Loader(
            DATASET.get_local_path() / f"{mode}/random-{mode}.beton",
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            distributed=(mode == "train") and (self.distributed),
            order=ffcv.loader.OrderOption.RANDOM,
            drop_last=(mode == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )
        return loader
