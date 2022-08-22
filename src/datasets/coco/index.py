from src.datasets.coco.hub3 import Hub3Dataset
from src.datasets.coco.pytorch import PytorchDataset
from src.datasets.coco.hub import HubDataset
from src.datasets.coco.squirrel import SquirrelDataset
from src.datasets.coco.torchdata import TorchdataDataset
from src.datasets.coco.webdataset import WebdatasetDataset

CocoDatasets = {
    "pytorch": PytorchDataset(),
    "hub": HubDataset(),
    "webdataset": WebdatasetDataset(),
    "torchdata": TorchdataDataset(),
    "hub3": Hub3Dataset(),
    "squirrel": SquirrelDataset(),
}