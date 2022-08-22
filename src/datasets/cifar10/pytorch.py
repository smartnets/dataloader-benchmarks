from src.datasets.base import Dataset
from src.datasets.cifar10.base import get_cifar10


class PytorchDataset(Dataset):

    def __init__(self):
        super().__init__("cifar10", "pytorch")

    def generate_locally(self, mode="train", transforms=None):
        get_cifar10(mode)

    def get_local(self, mode="train", transforms=None):

        return get_cifar10(
            mode=mode,
            download=False,
            transform=transforms
        )
