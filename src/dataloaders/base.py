from abc import ABC
from torch.utils.data.distributed import DistributedSampler
from distutils.util import strtobool
from src.utils.general import config_to_bool


class DataLoader(ABC):
    def __init__(
        self,
        filtering: bool = False,
        filtering_classes: list = None,
        automatic_gpu_transfer: bool = False,
        remote: bool = False,
        distributed: bool = False,
        world_size: int = None,
        rank: int = None,
    ):

        remote = config_to_bool(remote)

        self.automatic_gpu_transfer = automatic_gpu_transfer
        self.remote = remote
        self.filtering = filtering
        self.filtering_classes = filtering_classes
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank

    def get_train_loader(self, **kwargs):
        pass

    def get_val_loader(self, **kwargs):
        pass

    def get_test_loader(self, **kwargs):
        pass

    def get_distributd_sampler(self, ds):

        return DistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        )
