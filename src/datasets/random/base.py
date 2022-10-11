import torch
import time
from torchvision.datasets import CIFAR10
from pathlib import Path
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from src.config import settings as st

from src.datasets.transformations.to_tensor import to_tensor
from src.datasets.transformations.normalize import normalize
from src.datasets.transformations.cutout import cutout
from src.utils.distributed import get_worker_info


DATA_DIR = Path(st.local_data_dir)
DATA_DIR /= "random"
DATA_DIR /= "shared"

MEAN = (0.5, 0.5, 0.5)
STD = (0.2, 0.2, 0.2)

LABELS_DICT = dict((str(i), i) for i in range(20))
NUM_CLASASES = 20

def get_size(mode="train"):
    if mode == "test":
        return 500
    else:
        N = 50000
        VAL_SIZE = int(N * st.val_size)
        if mode == "train":
            return N - VAL_SIZE
        else:
            return VAL_SIZE

class RandomDataset(Dataset):
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

        self.root_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self._len()):
            if i % 100 == 0:
                print(f"{i} / {self._len()}", end="\r", flush=True)
            imarray = np.random.randint(low=0, high=256, size=(250, 250, 3), dtype=np.uint8)
            img = Image.fromarray(imarray, "RGB")
            path = self.root_dir / f"{i}.jpeg"
            img.save(path)

    def __len__(self):
        return self._len()

    def __getitem__(self, idx):

        path = self.root_dir / f"{idx}.jpeg"
        img = Image.open(path)


        if self.transform is not None:
            img = self.transform(img)

        return img, idx % NUM_CLASASES

    def __iter__(self):
        total_workers, global_worker_id = get_worker_info()
        for i in range(get_size(self.mode)):
            if i % total_workers == global_worker_id:
                yield self.__getitem__(i)



def get_random(mode="train", download=False, transform=None):

    is_train = mode in ["train", "val"]
    dataset = RandomDataset(DATA_DIR, mode, train=is_train,
                            download=download, transform=transform)

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

    # transform_list = [
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.ToTensor()
    #     normalize(np.array(MEAN), np.array(STD)),
    #     cutout(8, 1, False),
    #     to_tensor(),
    #     # Sleep()
    # ]
    # if to_pil:
    #     transform_list.insert(0, transforms.ToPILImage())

    transform_list = [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(MEAN, STD),
    ]
    
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
