import cv2
from myPackage import tools as tl
import numpy as np



def build_filters(size, theta, sigma, show= False):
    filters = []
    ksize = size
    for t in theta:
        for s in sigma:
            #cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
            kern = np.real(cv2.getGaborKernel((ksize, ksize), s, t, 10.0, 0.5, 0, ktype=cv2.CV_32F))
            kern /= 1.5 * kern.sum()
            filters.append(kern)

    if show:
        titles = []
        images = np.real(filters)
        title = 'Gabor kernels'
        tl.plotImages(titles, images, title, 6, 8)

    return filters


def convolve(img, filters, show= False):
    accum = np.zeros_like(img)
    for filter in filters:
        img_filt = cv2.filter2D(img, cv2.CV_8UC3, filter)
        # img_filt += img_filt
    np.maximum(accum, img_filt, accum)

    if show:
        titles = ['Original', 'Filtered']
        images = [img, img_filt]
        title = 'Apply Gabor filter'
        tl.plotImages(titles, images, title, 1, 2)

    return accum

