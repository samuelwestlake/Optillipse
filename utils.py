import cv2
import numpy as np


def load(filename, pad=0, invert=False):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im[im > 0] = 1
    if invert:
        im = 1 - im
    if pad > 0:
        h, w = im.shape
        im_ = np.zeros((h+pad*2, w+pad*2), dtype=np.uint8)
        im_[pad:pad+h, pad:pad+w] = im
        im = im_
    return im
