import cv2
import numpy as np


class Contour:
    def __init__(self, image):
        self.image = image
        self.contours, hierarchy = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def bounding_rect(self, contours=None):
        if contours is None:
            contours = self.contours
        x, y, w, h = cv2.boundingRect(contours)
        return int(x), int(y), int(w), int(h)


    def draw_bounding_rect(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        x, y, w, h = cv2.boundingRect(contours)
        return cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)


    def bounding_rect_rotated(self, contours=None):
            if contours is None:
                contours = self.contours
            rect = cv2.minAreaRect(contours)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return box

    def draw_bounding_rect_rotated(self, image=None, contours=None):
        if image is None:
            image = self.image
        if contours is None:
            contours = self.contours
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
        return img


    def moments(self, contours=None):
            if contours is None:
                contours = self.contours
            return cv2.moments(contours)

    # Center of Contours
    def center(self, contours=None):
        if contours is None:
            contours = self.contours
        M = cv2.moments(contours)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        contourCenter = (int(cx), int(cy))
        return contourCenter

    # Area of Contours
    def find_area(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.contourArea(contours)

    # Perimeter of Contours
    def perimeter(self, contours=None):
        if contours is None:
            contours = self.contours
        return cv2.arcLength(contours, True)

    # Contour Approximation
    def approximation(self, contours=None):
        if contours is None:
            contours = self.contours
        epsilon = 0.1 * cv2.arcLength(contours, True)
        approx = cv2.approxPolyDP(contours, epsilon, True)
        return epsilon, approx

    # Convex Hull
    # hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
    def hull(self, contours=None):
        if contours is None:
            contours = self.contours
            return cv2.convexHull(contours)
