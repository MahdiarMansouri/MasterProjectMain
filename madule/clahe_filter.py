import cv2 as cv
import numpy as np


class ClaheFilter:
    def __init__(self, image):
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.cl_img = clahe.apply(image)

    def get_cl_img(self):
        return self.cl_img

    def get_ravel(self):
        return np.ravel(self.cl_img)


# img = cv.imread('CV-FB.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# # create a CLAHE object (Arguments are optional).
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1 = clahe.apply(img)
# cv.imwrite('clahe_2.jpg', cl1)
# img = cv.imread("clahe_2.jpg")
# cv.imshow("01", img)
# cv.waitKey(0)
