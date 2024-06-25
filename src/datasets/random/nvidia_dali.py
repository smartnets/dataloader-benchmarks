from src.datasets.base import Dataset
from src.datasets.random.base import get_random
from torchvision import transforms
import os


class NvidiaDaliDataset(Dataset):

    def __init__(self):
        super().__init__("random", "nvidia_dali")

    def generate_locally(self, mode="train", transforms=None):
        """
        Convert RANDOM dataset to ImageFolder format
        """

        # datasets/random/nvidia_dali/train | val | test
        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        path.mkdir(parents=True, exist_ok=True)

        random = get_random(mode, download=True, transform=transforms)
        self.save_images(random, path)

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