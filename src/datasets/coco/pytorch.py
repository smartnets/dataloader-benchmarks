from src.datasets.base import Dataset
from src.datasets.coco.base import download_mode, get_coco


class PytorchDataset(Dataset):

    def __init__(self):
        super().__init__("coco", "pytorch")

    def generate_locally(self, mode="train", transforms=None):
        download_mode(mode)
        download_mode("annotations")

    def get_local(self, mode="train", transforms=None):

        dataset = get_coco(mode, transforms)

        return dataset
