# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import glob
import sklearn.metrics.pairwise

## import the handfeature extractor class
import handshape_feature_extractor
import frameextractor

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
pathlist = glob.glob(os.path.join("traindata",'*.mp4'))  #file paths for all mp4 files
features = np.zeros([len(pathlist),27])
frames_output = "frames"
hf = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance();

for i in range(len(pathlist)):
    frameextractor.frameExtractor(pathlist[i], frames_output, i)
    image = cv2.imread(os.path.join(frames_output, "%05d.png" % (i+1)), cv2.IMREAD_GRAYSCALE)
    features[i] = hf.extract_feature(image)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
test_frames_output = "test_output"
hf2 = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance();
testlist = glob.glob(os.path.join("test", '*.mp4'))
test_features = np.zeros([len(testlist),27])

for i in range(len(testlist)):
    frameextractor.frameExtractor(testlist[i],test_frames_output, i)
    image_2 = cv2.imread(os.path.join(test_frames_output, "%05d.png" % (i+1)), cv2.IMREAD_GRAYSCALE)
    test_features[i] = hf2.extract_feature(image_2)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

simil = features @ test_features.T
idx = np.argmax(simil, axis=0) // 3
np.savetxt("Results.csv", idx, delimiter=",", fmt='%d')

