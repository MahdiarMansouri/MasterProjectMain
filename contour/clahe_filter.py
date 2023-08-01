import cv2 as cv
import numpy as np


class clahe_filter:
    def __init__(self, image):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.cl_img = clahe.apply(image)

    def get_cl_img(self):
        return self.cl_img

    def get_ravel(self):
        return np.ravel(self.cl_img)

