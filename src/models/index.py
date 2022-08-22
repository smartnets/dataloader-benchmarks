from src.models.cnn import CNNModel
from src.models.faster_rcnn import FasterRCNNModel

MODEL_INDEX = {
    "cifar10": CNNModel,
    "random": CNNModel,
    "coco": FasterRCNNModel,
}
