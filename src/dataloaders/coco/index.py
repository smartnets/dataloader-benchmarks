from src.dataloaders.coco.pytorch import PytorchLoader
from src.dataloaders.coco.hub import HubLoader
from src.dataloaders.coco.hub3 import Hub3Loader
from src.dataloaders.coco.squirrel import SquirrelLoader
from src.dataloaders.coco.webdataset import WebdatasetLoader
from src.dataloaders.coco.torchdata import TorchdataLoader

CocoLoaders = {
    "pytorch": PytorchLoader,
    "hub": HubLoader,
    "hub3": Hub3Loader,
    "webdataset": WebdatasetLoader,
    "torchdata": TorchdataLoader,
    "squirrel": SquirrelLoader,
}
