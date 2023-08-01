import cv2
import numpy as np

image = cv2.imread("clahe_2.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (1, 1))

# threshold = np.mean(blur)
# print(90)
threshold = 110

ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
# cv2.imshow("Image", thresh)
# cv2.waitKey(0)
# exit()

cnt, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# hierarchy ==> [Next, Previous, First_Child, Parent]
# cnt = np.array(cnt)
cv2.drawContours(image, cnt, -1, (255, 0, 0), 2)
locs = []
i = 1
for c in cnt:
    # M = cv2.moments(c)
    # cx = int(M['m10'] / M['m00'])
    # cy = int(M['m01'] / M['m00'])
    # contourCenter = (int(cx), int(cy))
    #
    area = cv2.contourArea(c)
    # arc_length = cv2.arcLength(c, True)
    # print(area, arc_length)
    if area > 400:
        x, y, w, h = cv2.boundingRect(c)

        rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
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

        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # box = np.intp(box)
        # image = cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # image = cv2.drawContours(image, rect, 0, (0, 255, 0), 2)

        # todo : needs at least 5 point
        # ellipse = cv2.fitEllipse(c)
        # cv2.ellipse(image, ellipse, (0, 0, 255), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)
print(locs)
exit()
