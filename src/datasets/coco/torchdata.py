from src.datasets.base import Dataset
from src.datasets.coco.base import get_coco
from src.datasets.coco.base import download_mode, get_coco
# from torchdata.datapipes.map import SequenceWrapper
# from torchdata.datapipes.iter import IterableWrapper

from src.libraries.torchdata import build_dataset


class TorchdataDataset(Dataset):
    def __init__(self):
        super().__init__("coco", "torchdata")

    def generate_locally(self, mode="train", transforms=None):
        # Using the default pytorch method for getting cifar10
        download_mode(mode)
        download_mode("annotations")


    def generate_remotely(self, mode="train", transforms=None):
        pass

    def get_local(self, mode="train", transforms=None, distributed=False, batch_size=None):
        ds = get_coco(mode, transforms)
        return build_dataset(ds)

        # # NOTE: for Iterable-Style Dataset use:
        # ds = IterableWrapper(ds)

        # # Using MapDataPipe class
        # # ds = SequenceWrapper(random)
        # if distributed:
        #     ds = ds.sharding_filter()
        # return ds

    def get_remote(self, mode="train", transforms=None, distributed = False):
        pass
