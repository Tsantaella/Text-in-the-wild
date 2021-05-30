import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils.object_detection import non_max_suppression

import h5py

import scipy.io
import cv2

import numpy as np
import copy
import time
import pandas as pd
import os

def detectText(image, min_confidence=0.5):
    detector = cv2.dnn.readNet("frozen_east_text_detection.pb")

    orig = copy.deepcopy(image)

    (H, W) = image.shape[:2]

    newW = 640
    newH = 480

    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    detector.setInput(blob)
    (scores, geometry) = detector.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    #print("[INFO] text detection took {:.6f} seconds".format(end - start))
    #print("")
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x*4.0, y*4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    origText = copy.deepcopy(orig)
    bounds = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        bounds.append([startX, startY, endX, endY]) 
        # Dibujar el rectangulito
        cv2.rectangle(origText, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
    return origText, bounds