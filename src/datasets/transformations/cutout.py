import numpy as np
from src.datasets.transformations.base import Transformation


class cutout(Transformation):

    def __init__(self, mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0
        self.cutout_inside = cutout_inside
        self.mask_color = mask_color
        self.mask_size = mask_size
        self.p = p

    def __call__(self, image):
        image = np.asarray(image).copy()

        cutout_inside = self.cutout_inside
        mask_size_half = self.mask_size_half
        offset = self.offset
        mask_size = self.mask_size

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color
        return image
