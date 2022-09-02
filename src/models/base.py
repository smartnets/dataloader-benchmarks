from abc import ABC
import torch
from more_itertools import distribute
from src.utils.distributed import setup, cleanup
from src.dataloaders.random.pytorch import PytorchLoader
from src.profiling.metric_logger import MetricLogger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(ABC):
    def __init__(self, model, optimizer, criterion, metric, optimizer_args):
        self.model = model
        self.loss = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.optimizer_args = optimizer_args

    def initialize(self, rank, distributed: bool = False):

        model = self.model
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[rank],
                output_device=[device],
            )

        model.train()

        optimizer = self.optimizer(model.parameters(), lr=0.01)

        criterion = self.loss()
        # criterion = criterion.to(device)

        self.init_model = model
        self.init_optimizer = optimizer
        self.init_criterion = criterion
        self.device = device
