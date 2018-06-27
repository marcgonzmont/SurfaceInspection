import sys
from os.path import basename
import argparse
import numpy as np
import copy
import time
from myPackage import tools as tl
from myPackage import Gabor_filter as gf

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr", "--training_path", required=True,
                    help="-tr Training path of the samples")
    ap.add_argument("-te", "--test_path", required=True,
                    help="-te Test path of the samples")
    # ap.add_argument("-gt", "--gt_file", required=True,
    #                 help="-gt GT file to measure the performance of the algorithm")
    args = vars(ap.parse_args())


    # Get the training and test images to process and the GT to evaluate the algorithm
    training_images = tl.natSort(tl.getSamples(args["training_path"],0))
    test_images = tl.natSort(tl.getSamples(args["test_path"],0))
    gt_train = tl.natSort(tl.getSamples(args["training_path"],1))
    print(gt_train[0])
    data = tl.getGTyaml(gt_train[0])
    # for section in data:
    #     print(section)


    # frequency = np.arange(0, 1, 0.25)   #(0.05, 0.25)
    # theta = np.arange(0, np.pi, np.pi / 5)
    # sigma = np.arange(1, 4, 1)
    # size = 15
    #
    # training = False
    #
    # # if training:
    # #
    # # else:
    #
    # for tr in training_images:
    #     img = gf.cv2.imread(tr)
    #     a, b = gf.whitefill(img)
    #     img_copy = copy.deepcopy(img)
    #
    #     filters = gf.build_filters_cv(size= size, theta= theta, sigma= sigma)
    #     img_copy = gf.cv2.GaussianBlur(img_copy, (3,3), 0)
    #     conv = gf.convolve(img_copy, filters)
    #     conv = gf.cv2.GaussianBlur(conv, (3, 3), 0)
    #     # conv = gf.whitefill(conv)
    #     thresh1 = gf.cv2.threshold(conv, 0, 255, gf.cv2.THRESH_BINARY + gf.cv2.THRESH_OTSU)[1]
    #     thresh1 = gf.cv2.erode(thresh1, np.ones((7, 7)))
    #     thresh1[0:a+5, :] = 255
    #     thresh1[b-5:, :] = 255
    #
    #
    #     title = 'Result for {}'.format(basename(tr))
    #     titles = ['Original', 'Gabor filter']
    #     images = [img, thresh1]
    #     tl.plotImages(titles= titles, images= images, title= title, row= 1, col= 2)




    sys.exit(0)