import cv2
from madule.contour import Contour
from madule.image_madule import *
from madule.clahe_filter import ClaheFilter
import numpy as np


image2clahe = read_img_clahe("files/CV-FB.jpg")
image = read_img("files/CV-FB.jpg")
clahe = ClaheFilter(image2clahe)
cl_img = clahe.get_cl_img()

blur = cv2.blur(cl_img, (1, 1))

threshold = 120

ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

contour = Contour(thresh)
cnt = contour.contours
cv2.drawContours(image, cnt, -1, (255, 0, 0), 2)
locs = []
i = 1
for c in cnt:
    area = contour.find_area(contours=c)
    if area > 400:
        x, y, w, h = contour.bounding_rect(contours=c)
        rect = contour.draw_bounding_rect(image, contours=c)

        a = {
            "code": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        }
        locs.append(a)
        i = i + 1
        print(f"{x, y, w, h}")

cv2.imshow("Image", image)
cv2.waitKey(0)
print(locs)
