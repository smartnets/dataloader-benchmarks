import hub
import sys
import numpy as np
from src.datasets.base import Dataset
from src.datasets.coco.base import get_coco, LABEL_DICT
from indra import api
from src.utils.persist import get_s3_creds
from src.libraries.hub3 import create_dataset
from src.config import settings as st


class Hub3Dataset(Dataset):
    def __init__(self):
        super().__init__("coco", "hub3")

    def generate_locally(self, mode="train", transforms=None):

        path = super().generate_locally(mode, transforms)
        if not path:
            return None
        coco = get_coco(mode, None)
        ds = hub.empty(str(path), overwrite=True)
        return create_dataset(coco, ds, list(LABEL_DICT.keys()), "coco")

    def generate_remotely(self, mode="train", transforms=None):

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None
        coco = get_coco(mode, None)
        ds = hub.empty(str(path), creds=get_s3_creds(), overwrite=True)

        return create_dataset(coco, ds, list(LABEL_DICT.keys()), "coco")

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
