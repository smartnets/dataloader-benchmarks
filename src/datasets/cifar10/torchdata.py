from src.datasets.base import Dataset
from src.datasets.cifar10.base import get_cifar10
from src.libraries.torchdata import build_dataset


class TorchdataDataset(Dataset):
    def __init__(self):
        super().__init__("cifar10", "torchdata")

    def generate_locally(self, mode="train", transforms=None):
        # Using the default pytorch method for getting cifar10
        cifar = get_cifar10(mode, download=True, transform=transforms)

    def generate_remotely(self, mode="train", transforms=None):
        pass

    def get_local(self, mode="train", transforms=None, distributed: bool = False, batch_size: int = 8):
        cifar = get_cifar10(mode, download=False, transform=transforms)

        return build_dataset(cifar, distributed)


    def get_remote(self, mode="train", transforms=None, distributed: bool = False, batch_size: int = 8):
        pass
