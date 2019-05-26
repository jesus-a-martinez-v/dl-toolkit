import cv2
import numpy as np


class CropPreprocessor(object):
    def __init__(self, width, height, horizontal=True, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horizontal = horizontal
        self.interpolation = interpolation

    def preprocess(self, image):
        crops = list()

        h, w = image.shape[:2]

        coordinates = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]

        # Computes the center crop of the image.
        delta_w = int(0.5 * (w - self.width))
        delta_h = int(0.5 * (h - self.height))
        coordinates.append([delta_w, delta_h, w - delta_w, h - delta_h])

        for (start_x, start_y, end_x, end_y) in coordinates:
            crop = image[start_y:end_y, start_x:end_x]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.interpolation)
            crops.append(crop)

        if self.horizontal:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)
