import hub
import logging
import sys
import numpy as np
from src.datasets.base import Dataset
from src.datasets.coco.base import get_coco, LABEL_DICT
from src.config import settings as st
from src.utils.persist import get_s3_creds


class HubDataset(Dataset):
    def __init__(self):
        super().__init__("coco", "hub")

    def _create(self, dataset, ds):

        class_names = list(LABEL_DICT.keys())
        with ds:
            # Create the tensors with names of your choice.
            ds.create_tensor("images", htype="image",
                             sample_compression='jpeg')
            ds.create_tensor("labels", htype="class_label",
                             class_names=class_names)
            ds.create_tensor('boxes', htype='bbox')

            ds.boxes.info.update(coords={'type': 'fractional', 'mode': 'LTWH'})

        # %%
        with ds:
            for i, data in enumerate(dataset):
                image, targets = data
                if len(targets["categories"]) > 0:
                    ds.images.append(image)
                    ds.labels.append(targets["categories"])
                    ds.boxes.append(targets["boxes"])
                print(f"{i} / {len(dataset)}", flush=True, end="\r")

        return ds

    def generate_locally(self, mode="train", transforms=None):

        is_train = mode in ["train", "val"]
        path = self.get_local_path()
        path /= f"{mode}"

        coco = get_coco(mode, None)

        ds = hub.empty(str(path), overwrite=True)

        return self._create(coco, ds)

    def generate_remotely(self, mode="train", transforms=None):

        r_path = self.get_remote_path()
        r_path += f"/{mode}"

        coco = get_coco(mode, None)

        ds = hub.empty(str(r_path), creds=st.s3_creds, overwrite=True)

        return self._create(coco, ds)

    def get_local(self, mode="train", transforms=None):
        is_train = mode in ["train", "val"]

        path = self.get_local_path()
        path /= f"{mode}"

        ds = hub.load(str(path))
        return ds

    def get_remote(self, mode="train", transforms=None):
        r_path = self.get_remote_path()
        r_path += f"/{mode}"

        ds = hub.load(str(r_path), creds=get_s3_creds())
        return ds
