import cv2
from madule.contour import Contour
from madule.image_madule import *
from madule.clahe_filter import ClaheFilter


class ContourFinder:
    def __init__(self, path, img_number):
        image2clahe = read_img_clahe(path)
        self.image = read_img(path)
        clahe = ClaheFilter(image2clahe)
        cl_img = clahe.get_cl_img()

        blur = cv2.blur(cl_img, (1, 1))

        threshold = 120

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        self.contour = Contour(thresh)
        self.cnt = self.contour.contours
        cv2.drawContours(self.image, self.cnt, -1, (255, 0, 0), 2)
        self.locs = [img_number]

    def draw_contours(self):
        i = 1
        for c in self.cnt:
            area = self.contour.find_area(contours=c)
            if area > 400:
                x, y, w, h = self.contour.bounding_rect(contours=c)
                self.contour.draw_bounding_rect(self.image, contours=c)

                a = [i, x, y, w, h]

                self.locs.append(a)
                i = i + 1

        # cv2.imshow("Image", self.image)
        # cv2.waitKey(0)
        return self.locs
