from src.datasets.coco.deep_lake import DeepLakeDataset
from src.datasets.coco.pytorch import PytorchDataset
from src.datasets.coco.hub import HubDataset
from src.datasets.coco.squirrel import SquirrelDataset
from src.datasets.coco.torchdata import TorchdataDataset
from src.datasets.coco.webdataset import WebdatasetDataset
from src.datasets.coco.nvidia_dali import NvidiaDaliDataset

CocoDatasets = {
    "pytorch": PytorchDataset(),
    "hub": HubDataset(),
    "webdataset": WebdatasetDataset(),
    "torchdata": TorchdataDataset(),
    "deep_lake": DeepLakeDataset(),
    "squirrel": SquirrelDataset(),
    "nvidia_dali": NvidiaDaliDataset(),
}