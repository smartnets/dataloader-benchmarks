from abc import ABC
from pathlib import Path
from src.config import settings as st
from src.utils.persist import folder_exists_and_not_empty


class Dataset(ABC):
    def __init__(self, dataset, library):
        self.dataset = dataset
        self.library = library

    def get_local_path(self):
        return Path(st.local_data_dir) / self.dataset / self.library

    def get_remote_path(self):
        return (
            st.remote_data_dir.format(BUCKET_NAME=st.bucket_name)
            + "/"
            + self.dataset
            + "/"
            + self.library
        )

    def generate_locally(self, mode, transforms=None):
        path = self.get_local_path()
        path /= mode

        path = Path(path)
        if path.is_dir():
            print(
                f"Dataset {self.dataset} using {self.library} already exists in {path}"
            )
            return None
        else:
            return path

    def generate_remotely(self, mode, transforms=None):
        path = self.get_remote_path()
        path += f"/{mode}"

        s3_path = path.split(st.bucket_name + "/")[1]
        if folder_exists_and_not_empty(st.bucket_name, s3_path):
            print(
                f"Dataset {self.dataset} using {self.library} already exists in {s3_path}"
            )
            return None
        else:
            return path

    def get_local(
        self,
        transforms=None,
        filtering=False,
        filtering_classes=None,
        distributed=False,
    ):
        pass

    def get_remote(
        self,
        transforms=None,
        filtering=False,
        filtering_classes=None,
        distributed=False,
    ):
        pass
