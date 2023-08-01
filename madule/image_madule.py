import cv2 as cv
import numpy as np
import pandas as pd


def read_img(path):
    img = cv.imread(path)
    return img

def read_img_clahe(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img

def show_img(image):
    cv.imshow("Image", image)
    cv.waitKey(0)

def image2signal(img):
    return np.ravel(img)


def signal2image(signal, shape):
    return np.array(signal.reshape(shape))


def regression(X, y, degree=1):
    x = np.ravel(X)
    model = np.poly1d(np.polyfit(x, y, degree))
    return model(x)


def std_lines(y, reg, up_coef=1, down_coef=1):
    std = y.std()
    return reg + (up_coef * std), reg - (down_coef * std)


def make_dataframe(signal, columns_name):
    data = signal
    index_value = np.arange(len(signal))
    return pd.DataFrame(data=data, index=index_value, columns=columns_name)


def light_normalizer(img):
    y = image2signal(img)
    X = np.arange(len(y))
    reg = regression(X, y)
    mean = y.mean()
    y = y + (mean - reg)
    img = signal2image(y, img.shape)
    return img
