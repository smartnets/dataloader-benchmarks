import torch
import time
from torchvision.datasets import CIFAR10
from pathlib import Path
import numpy as np
from torchvision import transforms
from src.config import settings as st
from PIL import Image

from torch.utils.data import Dataset
from src.datasets.transformations.to_tensor import to_tensor
from src.datasets.transformations.normalize import normalize
from src.datasets.transformations.cutout import cutout
from src.utils.distributed import get_worker_info


DATA_DIR = Path(st.local_data_dir)
DATA_DIR /= "cifar10"
DATA_DIR /= "shared"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Class Names for Filtering
# "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
# See https://www.cs.toronto.edu/~kriz/cifar.html
LABELS_DICT = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

def get_size(mode="train"):
    if mode == "test":
        return 1000
    else:
        N = 50000
        VAL_SIZE = int(N * st.val_size)
        if mode == "train":
            return N - VAL_SIZE
        else:
            return VAL_SIZE

class Cifar10Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, mode, train=True, transform=None, download=False):
        self.root_dir = Path(root_dir) / mode
        self.mode = mode
        self.transform = transform
        self.is_train = train

        if download and not self.root_dir.is_dir():
            self._download()
        else:
            print("Already exists. Skipping")

    def _len(self):
        if self.is_train:
            return get_size("train") + get_size("val")
        else:
            return get_size("test")

    def _download(self):

        tmp_path = self.root_dir.parent / f"tmp_{'train' if self.is_train else 'test'}"
        ds = CIFAR10(tmp_path, train=self.is_train, download=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(ds):
            if i % 100 == 0:
                print(f"{i} / {self._len()}", end="\r", flush=True)

            img, cls = sample
            cls = str(cls)
            path = self.root_dir / f"{i}.jpeg"
            img.save(path)
            with open(path.with_suffix(".txt"), "w") as fh:
                fh.write(cls)


    def __len__(self):
        return self._len()

    def __getitem__(self, idx):

        path = self.root_dir / f"{idx}.jpeg"
        img = Image.open(path)
        with open(path.with_suffix(".txt"), "r") as fh:
            cls = int(fh.read())


        if self.transform is not None:
            img = self.transform(img)

        return img, cls

    def __iter__(self):
        total_workers, global_worker_id = get_worker_info()
        for i in range(get_size(self.mode)):
            if i % total_workers == global_worker_id:
                yield self.__getitem__(i)


def get_cifar10(mode="train", download=True, transform=None):

    is_train = mode in ["train", "val"]
    path = DATA_DIR
    if is_train:
        path /= "train"
    else:
        path /= "test"

    dataset = Cifar10Dataset(DATA_DIR, mode, train=is_train, download=download, transform=transform)

    gen = torch.Generator()
    gen.manual_seed(0)

    if is_train:
        ds_train, ds_val = torch.utils.data.random_split(
            dataset, [get_size("train"), get_size("val")], generator=gen
        )
        if mode == "train":
            return ds_train
        elif mode == "val":
            return ds_val
    else:
        return dataset


def get_train_transforms(to_pil: bool = False):

    transform_list = [
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor()
        normalize(np.array(MEAN), np.array(STD)),
        cutout(8, 1, False),
        to_tensor(),
        # Sleep()
    ]
    if to_pil:
        transform_list.insert(0, transforms.ToPILImage())

    train_transform = transforms.Compose(transform_list)
    return train_transform


def get_eval_transforms():
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return eval_transform
