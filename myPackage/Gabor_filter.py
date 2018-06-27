import cv2
from numba import jit
import numpy as np
from matplotlib import pyplot as plt
# import copy
# from skimage.util import img_as_float
from skimage.filters import gabor_kernel

@jit(parallel= True)
def build_filters_sk(theta, sigma, frequency):
    filters = []
    for t in theta:
        for s in sigma:
            for f in frequency:
                filter = np.real(gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s))
                filters.append(filter)
    return filters

# @jit(parallel= True)
def build_filters_cv(size, theta, sigma):
    filters = []
    ksize = size
    for t in theta:
        for s in sigma:
            #cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
            kern = cv2.getGaborKernel((ksize, ksize), s, t, 9, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    return filters

@jit(parallel= True)
def convolve(img, filters):
    accum = np.zeros_like(img)
    for filter in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, filter)
        np.maximum(accum, fimg, accum)
    return accum

def whitefill(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, (5,5), iterations= 3)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    print(x, y, w, h)
    # rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    # (x, y), (w, h) = rect
    # print(rect)

    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # print([box])
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # cv2.rectangle(img, (x,y+5), (x+w,y+h-5),(255,0,0))
    # roi = thresh[y:y + h, x:x + w]
    # plt.imshow(thresh, cmap= 'gray')
    # plt.show()

    return y+5, y+h-10
