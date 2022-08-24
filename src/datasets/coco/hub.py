import hub
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

        path = super().generate_locally(mode, transforms)
        if not path:
            return None

        coco = get_coco(mode, None)

        ds = hub.empty(str(path), overwrite=True)

        return self._create(coco, ds)

    def generate_remotely(self, mode="train", transforms=None):

        path = super().generate_remotely(mode, transforms)
        if not path:
            return None

        coco = get_coco(mode, None)

        ds = hub.empty(str(path), creds=get_s3_creds(), overwrite=True)

        return self._create(coco, ds)

    def get_local(self, mode="train", transforms=None):
        path = self.get_local_path()
        path /= f"{mode}"
        return  hub.load(str(path))

    def get_remote(self, mode="train", transforms=None):
        r_path = self.get_remote_path()
        r_path += f"/{mode}"
        return  hub.load(str(r_path), creds=get_s3_creds())
