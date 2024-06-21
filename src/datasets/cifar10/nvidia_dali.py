# Adapted from: https://github.com/NVIDIA/DALI/blob/main/docs/examples/use_cases/pytorch/resnet50/main.py
from src.datasets.base import Dataset
from src.datasets.cifar10.base import get_cifar10
from torchvision import transforms
import os


class NvidiaDaliDataset(Dataset):

    def __init__(self):
        super().__init__("cifar10", "nvidia_dali")

    def generate_locally(self, mode="train", transforms=None):
        """
        Convert CIFAR10 dataset to ImageFolder format
        """

        # datasets/cifar10/nvidia_dali/train | val | test
        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True, exist_ok=True)

        cifar = get_cifar10(mode, download=True, transform=transforms)
        self.save_images(cifar, path)

    def generate_remotely(self, mode="train", transforms=None):
        pass

    def get_local(self, transforms=None):
        """
        Does not seem like Nvidia DALI has a get dataset 
        """
        pass

    def get_remote(self, transforms=None):
        pass

    def save_images(self, dataset, root_dir):
        """
        Convert the dataset to ImageFolder format.
        """
        for idx, (image, label) in enumerate(dataset):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            img_path = os.path.join(class_dir, f'{idx}.png')
            image.save(img_path)