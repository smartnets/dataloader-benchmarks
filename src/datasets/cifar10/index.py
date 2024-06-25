from src.datasets.cifar10.ffcv import FFCVDataset
from src.datasets.cifar10.deep_lake import DeepLakeDataset
from src.datasets.cifar10.pytorch import PytorchDataset
from src.datasets.cifar10.hub import HubDataset
from src.datasets.cifar10.torchdata import TorchdataDataset
from src.datasets.cifar10.webdataset import WebdatasetDataset
from src.datasets.cifar10.squirrel import SquirrelDataset
from src.datasets.cifar10.nvidia_dali import NvidiaDaliDataset

CIFAR10Datasets = {
    "ffcv": FFCVDataset(),
    "pytorch": PytorchDataset(),
    "hub": HubDataset(),
    "webdataset": WebdatasetDataset(),
    "torchdata": TorchdataDataset(),
    "squirrel": SquirrelDataset(),
    "deep_lake": DeepLakeDataset(),
    "nvidia_dali": NvidiaDaliDataset(),
}