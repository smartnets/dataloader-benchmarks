import logging
import sys
import boto3
from botocore.client import Config
import numpy as np
from ...libraries.webdataset import build_dataset
from src.utils.persist import save_results_s3
import webdataset as wds
from src.datasets.base import Dataset
from src.datasets.coco.base import get_coco
import torchvision.transforms
from src.utils.persist import save_results_s3
import webdataset as wds


class WebdatasetDataset(Dataset):
    def __init__(self):
        super().__init__("coco", "webdataset")

        self.shard_prefix = "shard-%06d.tar"
        # self.num_shards = {  # These are manually hardcoded after looking at how many shards are created
        #     "train": 23,
        #     "test": 0,
        #     "val": 1,
        # }

    def generate_locally(self, mode="train", transforms=None):

        is_train = mode in ["train", "val"]
        path = self.get_local_path()
        path /= f"{mode}"
        if path.is_dir():
            print(f"{mode} already downloaded.")
            return None
        path.mkdir(parents=True)

        sink = wds.ShardWriter(str(path / self.shard_prefix), maxcount=5000)
        # cifar = get_cifar10(mode, download=True, transform=transforms)
        coco = get_coco(mode, None)

        for index, (image, target) in enumerate(coco):
            if index % 1000 == 0:
                print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
            sink.write(
                {
                    "__key__": "sample%06d" % index,
                    "image.jpg": image,
                    "labels.npy": np.array(target["categories"]),
                    "boxes.npy": np.array(target["boxes"]),
                }
            )
        sink.close()

    def _remote_name(self, shard, mode):

        return f"datasets/{self.dataset}/{self.library}/{mode}/{self.shard_prefix}" % shard

    def generate_remotely(self, mode="train", transforms=None):
        self.generate_locally(mode=mode, transforms=transforms)

        path = self.get_remote_path()
        path += f"{mode}"
        num_shards = self.get_num_shards(mode)

        for i in range(num_shards):
            name = self.shard_prefix % i

            local_path = self.get_local_path()
            local_path /= f"{mode}"
            local_path /= name

            remote_name = self._remote_name(i, mode)
            save_results_s3(str(local_path), remote_name)

    def get_num_shards(self, mode):
        path = self.get_local_path()
        path /= f"{mode}"
        num_shards = sum(1 for _ in path.glob("*.tar"))
        return num_shards

    def get_local(self, mode="train", transforms=None, distributed: bool = False, filtering: bool = False, filtering_classes: list = [], batch_size: int = 4, shuffle: bool = False):

        path = self.get_local_path()
        path /= f"{mode}"

        num_shards = self.get_num_shards(mode)
        path /= "shard-{%06d..%06d}.tar" % (0, num_shards - 1)
        path = str(path)

        FC = []

        return build_dataset(
            path,
            distributed,
            transforms,
            filtering,
            FC,
            batch_size,
            shuffle,
            "coco")

    def get_remote(self, mode="train", transforms=None, distributed: bool = False, filtering: bool = False, filtering_classes: list = [], batch_size: int = 4, shuffle: bool = False):


        num_shards = self.get_num_shards(mode)
        path = self.get_remote_path()
        path += f"/{mode}"
        path += "/shard-{%06d..%06d}.tar -" % (0, num_shards)
        path = "pipe:s3cmd -q get --force " + path

        FC = []

        return build_dataset(
            path,
            distributed,
            transforms,
            filtering,
            FC,
            batch_size,
            shuffle,
            "coco")

