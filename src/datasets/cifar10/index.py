from src.datasets.cifar10.ffcv import FFCVDataset
from src.datasets.cifar10.hub3 import Hub3Dataset
from src.datasets.cifar10.pytorch import PytorchDataset
from src.datasets.cifar10.hub import HubDataset
from src.datasets.cifar10.torchdata import TorchdataDataset
from src.datasets.cifar10.webdataset import WebdatasetDataset
from src.datasets.cifar10.squirrel import SquirrelDataset

CIFAR10Datasets = {
    "ffcv": FFCVDataset(),
    "pytorch": PytorchDataset(),
    "hub": HubDataset(),
    "webdataset": WebdatasetDataset(),
    "torchdata": TorchdataDataset(),
    "squirrel": SquirrelDataset(),
    "hub3": Hub3Dataset(),
}