import cv2
import imutils


class AspectAwarePreprocessor(object):
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = inter

    def preprocess(self, image):
        h, w = image.shape[:2]
        delta_w = 0
        delta_h = 0

        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.interpolation)
            delta_h = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.interpolation)
            delta_w = int((image.shape[1] - self.width) / 2.0)

        h, w = image.shape[:2]
        image = image[delta_h:h - delta_h, delta_w: w - delta_w]

        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
