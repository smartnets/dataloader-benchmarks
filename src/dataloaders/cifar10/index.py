from src.dataloaders.cifar10.ffcv import FFCVLoader
from src.dataloaders.cifar10.deep_lake import DeepLakeLoader
from src.dataloaders.cifar10.pytorch import PytorchLoader
from src.dataloaders.cifar10.hub import HubLoader
from src.dataloaders.cifar10.squirrel import SquirrelLoader
from src.dataloaders.cifar10.webdataset import WebdatasetLoader
from src.dataloaders.cifar10.torchdata import TorchdataLoader
from src.dataloaders.cifar10.nvidia_dali import NvidiaDaliLoader

CIFAR10Loaders = {
    "pytorch": PytorchLoader,
    "ffcv": FFCVLoader,
    "hub": HubLoader,
    "webdataset": WebdatasetLoader,
    "torchdata": TorchdataLoader,
    "squirrel": SquirrelLoader,
    "deep_lake": DeepLakeLoader,
    "nvidia_dali": NvidiaDaliLoader,
}
