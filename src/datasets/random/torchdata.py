from src.datasets.base import Dataset
from src.datasets.random.base import get_random

from src.libraries.torchdata import build_dataset


class TorchdataDataset(Dataset):
    def __init__(self):
        super().__init__("random", "torchdata")

    def generate_locally(self, mode="train", transforms=None):
        # Using the default pytorch method for getting cifar10
        random = get_random(mode, download=True, transform=transforms)

    def generate_remotely(self, mode="train", transforms=None):
        pass

    def get_local(self, mode="train", transforms=None, distributed=False, batch_size=None):
        random = get_random(mode, download=False, transform=transforms)
        return build_dataset(random, distributed)

    def get_remote(self, mode="train", transforms=None, distributed = False):
        pass
