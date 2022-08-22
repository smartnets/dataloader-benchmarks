from src.datasets.transformations.base import Transformation

class Identity(Transformation):
    def __call__(self, x):
        return x