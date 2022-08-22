import numpy as np
from src.datasets.transformations.base import Transformation

class normalize(Transformation):

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = np.asarray(image).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image