import hub
import logging
import sys
import numpy as np
from src.datasets.base import Dataset
from src.config import settings as st
from src.datasets.cifar10.base import get_cifar10
from indra import api, Loader

from src.utils.persist import get_s3_creds


# Probably, hub3 loader will be able to work with
# datasets created by hub, so Hub3Dataset will be
# the same as HubDataset


class Hub3Dataset(Dataset):
    def __init__(self):
        super().__init__("cifar10", "hub3")

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
            ds.create_tensor("images", htype="image", sample_compression="jpeg")
            ds.create_tensor("labels", htype="class_label", class_names=class_names)

            # Add arbitrary metadata - Optional
            ds.info.update(description="CIFAR 10")
            ds.images.info.update(camera_type="SLR")
        with ds:
            for i, (image, label) in enumerate(cifar):
                print(f"Iteration {i:4d}", end="\r", flush=True, file=sys.stderr)
                ds.append({"images": np.array(image), "labels": np.uint8(label)})

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

        ds = hub.empty(str(path), overwrite=True, creds=get_s3_creds())

        return self._create(cifar, ds)

    def get_local(self, mode="train", transforms=None):
        path = self.get_local_path()
        path /= f"{mode}"
        return api.dataset(str(path))

    def get_remote(self, mode="train", transforms=None):
        r_path = self.get_remote_path()
        r_path += f"/{mode}"
        creds = get_s3_creds()
        if creds["endpoint_url"] is None:
            del creds["endpoint_url"]
        return api.dataset(str(r_path), **creds)
