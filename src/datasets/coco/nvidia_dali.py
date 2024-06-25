from src.datasets.base import Dataset
from src.datasets.coco.base import download_mode, get_coco


# Adapted from: https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
from torchvision import transforms
import os


class NvidiaDaliDataset(Dataset):

    def __init__(self):
        super().__init__("coco", "nvidia_dali")

    def generate_locally(self, mode="train", transforms=None):
        """
        Convert COCO dataset to ImageFolder format
        """

        # if not datasets/coco/shared is downloaded, download it
        # if not os.path.exists("datasets/coco/shared"):
        #     download_mode(mode)
        #     download_mode("annotations")
        # # # datasets/coco/nvidia_dali/train | val | test
        # # path = super().generate_locally(mode, transforms)
        # # if not path:
        # #     return None
        # # path.mkdir(parents=True, exist_ok=True)

        coco = get_coco(mode, download=True, transform=transforms)

    def generate_remotely(self, mode="train", transforms=None):
        pass

    def get_local(self, transforms=None):
        """
        Does not seem like Nvidia DALI has a get dataset 
        """
        pass

    def get_remote(self, transforms=None):
        pass