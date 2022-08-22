from src.datasets.transformations.base import Transformation

class CustomTransform(Transformation):

    def __init__(self, transforms, distributed):
        self.transforms = transforms
        self.distributed = distributed

    def __call__(self, value):
        if self.distributed:
            return {"images": self.transforms(value["jpg"]), "labels": value["cls"]}
        else:
            return (self.transforms(value["jpg"]), value["cls"])