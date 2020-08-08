#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sample demonstrates structured edge detection and edgeboxes.
Usage:
  edgeboxes_demo.py [<model>] [<input_image>]
'''

import cv2 as cv
import numpy as np
from PIL import Image


def filtering(param, index):
    if param[1] in index:
        return True
    return False


def edgeBoxes(model, image_param):
    print(__doc__)
    returned_boxes = []  # The boxes that we will return because they achieved high score.
    image_bgr = cv.imread(image_param)
    image_bgr = cv.resize(image_bgr, (224, 224))
    edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)  # Edge_boxes detection and scores.
    rgb_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_image) / 255.0)
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)
    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(100)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    boxes = list(filter(lambda param: param[2] * param[3] < 5000, boxes))
    boxes = boxes[:100]

    boxes = list(boxes)
    # scores = list(scores)
    # for j in range(boxes.__len__()):
    #     boxes[j] = boxes[j].tolist()
    # for j in range(scores.__len__()):
    #     scores[j] = float(scores[j][0])
    # indices = cv.dnn.NMSBoxes(boxes, scores, 0, 0.1)  # NMSBoxes returns the indices of the remaining boxes (IoU)
    # minTen = min(indices.__len__(), 10)  # We keep a maximum of 10 boxes
    # indices = indices[0:minTen]
    # for i in range(boxes.__len__()):
    #     boxes[i] = (boxes[i], i)
    #     scores[i] = (scores[i], i)
    # boxes = list(filter(lambda param: filtering(param, indices), boxes))
    # scores = list(filter(lambda param: filtering(param, indices), scores))
    # for i in range(boxes.__len__()):
    #     boxes[i] = boxes[i][0]
    #     scores[i] = scores[i][0]
    #
    # scores[:] = [x / sum(scores) for x in scores]  # Normalization
    # indices = cv.dnn.NMSBoxes(boxes, scores, 1 / minTen, 1)  # We might need to change the score factor.
    #
    # for i in range(boxes.__len__()):  # Second iteration. Now we keep the boxes that have a score higher than the factor
    #     boxes[i] = (boxes[i], i)
    #     scores[i] = (scores[i], i)
    # boxes = list(filter(lambda param: filtering(param, indices), boxes))
    # scores = list(filter(lambda param: filtering(param, indices), scores))
    # for i in range(boxes.__len__()):
    #     boxes[i] = boxes[i][0]
    #     scores[i] = scores[i][0]

    to_crop = Image.open(image_param)  # This parameter is used only to crop the wanted boxes from the original image.
    to_crop = to_crop.resize((231, 231))
    if len(boxes) > 0:
        boxes_scores = zip(boxes, boxes)
        for b_s in boxes_scores:
            box = b_s[0]
            x, y, w, h = box
            cropped = to_crop.crop((x, y, x + w, y + h))
            cropped = cropped.resize((231, 231))
            returned_boxes.insert(returned_boxes.__len__(), (cropped, w, h))
            # cropped.show()  # This is used only to print the cropped boxes and their scores ..
            # cv.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1, cv.LINE_AA)
            # score = b_s[1]
            # cv.putText
            # (image_bgr, "{:.2f}".format(score), (x, y), cv.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv.LINE_AA)
            # print("Box at (x,y)=({:d},{:d}); score={:f}".format(x, y, score))
    cv.destroyAllWindows()
    return returned_boxes
