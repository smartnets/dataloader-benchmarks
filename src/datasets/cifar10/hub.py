import hub
import sys
import numpy as np
from src.datasets.base import Dataset
from src.datasets.cifar10.base import get_cifar10
from src.config import settings as st
from src.utils.persist import get_s3_creds


class HubDataset(Dataset):
    def __init__(self):
        super().__init__("cifar10", "hub")

    def _create(self, cifar, ds):

        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        with ds:
            # Create the tensors with names of your choice.
            ds.create_tensor("images", htype="image",
                             sample_compression="jpeg")
            ds.create_tensor("labels", htype="class_label",
                             class_names=class_names)

            # Add arbitrary metadata - Optional
            ds.info.update(description="CIFAR 10")
            ds.images.info.update(camera_type="SLR")
        with ds:
            for i, (image, label) in enumerate(cifar):
                print(f"Iteration {i:4d}", end="\r",
                      flush=True, file=sys.stderr)
                ds.append({"images": np.array(image),
                          "labels": np.uint8(label)})

        return ds

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None

        cifar = get_cifar10(mode, download=True, transform=transforms)

        ds = hub.empty(str(path), overwrite=True)

        return self._create(cifar, ds)

    def generate_remotely(self, mode="train", transforms=None):

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None

        cifar = get_cifar10(mode, download=True, transform=transforms)

        ds = hub.empty(str(path), creds=get_s3_creds(), overwrite=True)

        return self._create(cifar, ds)

    def get_local(self, mode="train", transforms=None):
        is_train = mode in ["train", "val"]

        path = self.get_local_path()
        path /= f"{mode}"

        ds = hub.load(str(path))
        return ds

    def get_remote(self, mode="train", transforms=None):
        is_train = mode in ["train", "val"]
        r_path = self.get_remote_path()
        r_path += f"/{mode}"

        ds = hub.load(str(r_path), creds=get_s3_creds())
        return ds
