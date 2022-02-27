from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='book2.jpg')
parser.add_argument('--input2', help='Path to input image 2.', default='book_in_scene2.jpg')
args = parser.parse_args()

img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

orb = cv.ORB_create()
kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
knn_matches = matcher.knnMatch(des1, des2, 2)

ratio_thresh = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Good Matches', img_matches)
cv.waitKey()
