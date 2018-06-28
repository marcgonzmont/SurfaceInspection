import sys
from os.path import basename
import argparse
import numpy as np
import glob
import cv2
from myPackage import tools as tl
from myPackage import image_processing as imp
from myPackage import Gabor_filter as gf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-tr", "--training_path",
                    help="-tr Training path of the samples")
    group.add_argument("-te", "--test_path",
                    help="-te Test path of the samples")
    args = vars(parser.parse_args())
    # Show intermediate images
    show = False
    debug = False
    # Margin to ensure to get the wood without bg
    margin = 15
    # Gabor filter configuration
    theta = np.arange(0, np.pi, np.pi / 16)          # orientation of the normal to the parallel stripes of the Gabor function
    sigma = np.arange(2, 5, 1)                      # controls the width of Gaussian envelope used in Gabor kernel
    size = 3                                       # the size of convolution kernel varies
    prev_file = ''
    results = []

    if args["training_path"] is not None:
        # Get the TRAINING and test images to process and the GT to evaluate the algorithm
        all_files = tl.natSort(tl.getSamples(args["training_path"]))
        text = ("\n--- TRAINIG RESULTS ---\n"
          "Intersection over Union (mean): {}\n"
          "True rejected: {}\n"
          "False rejected: {}\n"
          "False accepted: {}\n"
                "True accepted: {}\n")
    else:
        # Get the TEST images to process
        all_files = tl.natSort(tl.getSamples(args["test_path"]))
        text = ("\n--- TEST RESULTS ---\n"
                "Intersection over Union (mean): {}\n"
                "True rejected: {}\n"
                "False rejected: {}\n"
                "False accepted: {}\n"
                "True accepted: {}\n")

    for idx in range(len(all_files)):
        substr = '..' + all_files[idx].split(".")[-2]
        # Read all files except .directory
        if substr != prev_file and substr.split("/")[-1] != '':
            file = substr.split("/")[-1] + '.png'
            # print("Image '{}'".format(file))
            img, gt = tl.parseSample(substr)
            img_fg, coords = imp.removeBG(img.copy(), margin)       #debug
            filters = gf.build_filters(size, theta, sigma)          #debug
            img_filtered = gf.convolve(img_fg, filters)             #debug
            defects_detected = imp.detectDefects(img_filtered, debug)      #debug
            defects_detected = imp.getOriginalCoords(defects_detected, coords, margin)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Draw GT
            if gt is not None:
                for x, y, w, h in gt:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw detections
            if defects_detected is not None:
                for x, y, w, h in defects_detected:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if show:
                titles = []
                images = [img]
                title = "Detection result for '{}'".format(file)
                tl.plotImages(titles, images, title, 1, 1)
            results.append(imp.evaluate(gt, defects_detected))          # debug


        prev_file = substr
        # break
    results = np.array(results)

    iou_avg = np.mean([np.mean(iou) for iou in results[:, 0]])
    TR = np.sum(results[:, 1]).astype(int)
    FR = np.sum(results[:, 2]).astype(int)
    FA = np.sum(results[:, 3]).astype(int)
    TA = np.sum(results[:, 4]).astype(int)
    print(text.format(iou_avg, TR, FR, FA, TA))


    classes = ['rejected', 'accepted']
    cnf_matrix = np.zeros((2,2), dtype= int)
    cnf_matrix[0][0] = TR
    cnf_matrix[0][1] = FR
    cnf_matrix[1][0] = FA
    cnf_matrix[1][1] = TA

    tl.computeMetrics(cnf_matrix)

    np.set_printoptions(precision=3)
    # Plot normalized confusion matrix
    tl.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                             title="Normalized confusion matrix")


    print("\n\n---- FINISHED!! ----")
    sys.exit(0)