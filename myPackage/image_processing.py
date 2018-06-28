import cv2
from myPackage import tools as tl
import numpy as np


def removeBG(img, margin, show= False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(contours) != 0:
        # find the biggest area
        cnt_wood = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt_wood)
        fg = img[y + margin:y + h - margin, x:x + w]
    if show:
        copy = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
        titles = ['Contour', 'ROI']
        images = [copy, fg]
        title = 'Remove background'
        tl.plotImages(titles, images, title, 1, 2)

    return fg, (x, y, w, h)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def detectDefects(img_filtered, show= False):
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)
    defects_detected_nms = None
    copy = img_filtered.copy()
    # Threshold image
    thresh = cv2.threshold(copy, 65, 220, cv2.THRESH_BINARY_INV)[1]
    # Clean image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations= 3)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations= 2)
    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > 10 and cv2.boundingRect(cnt)[3] > 10]
    # filtered_contours = []
    # for cnt in contours:
    #     if cnt[2] > 10 and cnt[3] > 10:
    #         filtered_contours.append(cnt)

    if len(contours) != 0:
        # Get all bboxes
        defects_detected = [cv2.boundingRect(contour) for contour in contours]
        # Apply NMS to all bboxes
        defects_detected_nms = non_max_suppression_fast(np.array(defects_detected), 0.1)

        if show:
            copy_th = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
            # copy_th = cv2.drawContours(copy_th, contours, -1, (0, 255, 0), 2)
            for (x, y, w, h) in defects_detected_nms:
                cv2.rectangle(copy_th, (x, y), (x + w, y + h), (255, 0, 0), 2)

            titles = ['Filtered', 'Thresh (bbox)']
            images = [img_filtered, copy_th]
            title = 'Detect defects'
            tl.plotImages(titles, images, title, 1, 2)

    return  defects_detected_nms

def getOriginalCoords(defects_detected, coords_fg, margin):

    detections = []
    if defects_detected is not None:
        x_fg, y_fg = coords_fg[:2]

        for x, y, w, h in defects_detected:
            x_org = x + x_fg
            y_org = y + y_fg + margin
            detections.append([x_org, y_org, w, h])

    return detections


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if iou <= 0:
        iou = 0
    # return the intersection over union value
    return iou


def evaluate(gt, detections, show= False):
    '''
    Compute Intersection over Union and set True Rejected, False Rejected and False Accepted
    :param labels:
    :param detections:
    :return:
    '''
    iou = 0
    TR = 0
    FR = 0
    FA = 0
    TA = 0

    if gt is not None and detections is not None and len(detections) > 0:
        all_iou = np.zeros((len(gt), len(detections)))
        for i, box_gt in enumerate(gt):
            for j, detection in enumerate(detections):
                all_iou[i, j] = bb_intersection_over_union(box_gt, np.array(detection))
        iou = np.array([np.mean(label_iou) for label_iou in all_iou])
        iou = np.mean(iou)
        # TR = 1
        if iou > 0:
            TR = 1
        # else:
        #     FA = 1
    elif gt is not None and len(detections) == 0:
        FA = 1
    elif gt is None and len(detections) != 0:
        FR = 1
    elif gt is None and len(detections) == 0:
        TA = 1

    if show:
        print("IOU: {}\n"
              "TR: {}\n"
              "FR: {}\n"
              "FA: {}\n"
              "TA: {}\n".format(str(iou), str(TR), str(FR), str(FA), str(TA)))
    return iou, TR, FR, FA, TA