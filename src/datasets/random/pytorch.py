from pathlib import Path
from src.datasets.base import Dataset
from src.datasets.random.base import get_random


class PytorchDataset(Dataset):

    def __init__(self):
        super().__init__("random", "pytorch")

    def generate_locally(self, mode="train", transforms=None):
        get_random(mode, download=True)

    def get_local(self, mode="train", transforms=None):

        return get_random(
            mode=mode,
            download=False,
            transform=transforms
        )
