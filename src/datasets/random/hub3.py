import hub
import sys
import numpy as np
from src.datasets.base import Dataset
from src.datasets.random.base import get_random, LABELS_DICT
from indra import api
from src.utils.persist import get_s3_creds
from src.libraries.hub3 import create_dataset
from src.config import settings as st


class Hub3Dataset(Dataset):
    def __init__(self):
        super().__init__("random", "hub3")

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        random = get_random(mode, download=True, transform=transforms)
        ds = hub.empty(str(path), overwrite=True)
        return create_dataset(random, ds, list(LABELS_DICT.keys()))

    def generate_remotely(self, mode="train", transforms=None):

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None
        random = get_random(mode, download=True, transform=transforms)
        ds = hub.empty(str(path), creds=get_s3_creds(), overwrite=True)

        return create_dataset(random, ds, list(LABELS_DICT.keys()))

    def get_local(self, mode="train", transforms=None):
        path = self.get_local_path()
        path /= f"{mode}"
        return api.dataset(str(path))

    def get_remote(self, mode="train", transforms=None):
        path = self.get_remote_path()
        path += f"/{mode}"
        creds = get_s3_creds()

        if creds["endpoint_url"] is None:
            del creds["endpoint_url"]
        return api.dataset(str(path), **creds)
