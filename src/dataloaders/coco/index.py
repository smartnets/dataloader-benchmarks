from src.dataloaders.coco.pytorch import PytorchLoader
from src.dataloaders.coco.hub import HubLoader
from src.dataloaders.coco.deep_lake import DeepLakeLoader
from src.dataloaders.coco.squirrel import SquirrelLoader
from src.dataloaders.coco.webdataset import WebdatasetLoader
from src.dataloaders.coco.torchdata import TorchdataLoader

CocoLoaders = {
    "pytorch": PytorchLoader,
    "hub": HubLoader,
    "deep_lake": DeepLakeLoader,
    "webdataset": WebdatasetLoader,
    "torchdata": TorchdataLoader,
    "squirrel": SquirrelLoader,
}
