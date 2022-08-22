from src.dataloaders.cifar10.ffcv import FFCVLoader
from src.dataloaders.cifar10.hub3 import Hub3Loader
from src.dataloaders.cifar10.pytorch import PytorchLoader
from src.dataloaders.cifar10.hub import HubLoader
from src.dataloaders.cifar10.squirrel import SquirrelLoader
from src.dataloaders.cifar10.webdataset import WebdatasetLoader
from src.dataloaders.cifar10.torchdata import TorchdataLoader

CIFAR10Loaders = {
    "pytorch": PytorchLoader,
    "ffcv": FFCVLoader,
    "hub": HubLoader,
    "webdataset": WebdatasetLoader,
    "torchdata": TorchdataLoader,
    "squirrel": SquirrelLoader,
    "hub3": Hub3Loader,
}
