from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from numpy.lib.scimath import sqrt as npsqrt


# Input parameters
v = 4  # m/s
d = 20  # m
t = 1 / 30  # s
cx = 360  # pixels
cy = 480  # pixels
fx = 1800  # pixels
fy = 1800  # pixels
Z = 19  # m
beta = 0.57

parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN.')
# parser.add_argument('--input1', help='Path to input image 1.', default='book2.jpg')
# parser.add_argument('--input2', help='Path to input image 2.', default='book_in_scene2.jpg')
parser.add_argument('--input1', help='Path to input image 1.', default='stolb1.jpg')
parser.add_argument('--input2', help='Path to input image 2.', default='stolb2.jpg')
args = parser.parse_args()

img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
img1 = cv.resize(img1, (720, 960))
img2 = cv.resize(img2, (720, 960))


def detect_feature_points(image1, image2):
    if image1 is None or image2 is None:
        print('Could not open or find the images!')
        exit(0)

    orb = cv.ORB_create()
    kp1 = orb.detect(image1, None)
    kp2 = orb.detect(image2, None)

    kp1, des1 = orb.compute(image1, kp1)
    kp2, des2 = orb.compute(image2, kp2)

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    ratio_thresh = 0.7
    good_matches = []
    list_kp1 = []
    list_kp2 = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

            img1_idx = m.queryIdx
            img2_idx = m.trainIdx

            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))

    # print('k1 =', list_kp1)
    # print('k2 =', list_kp2)
    #
    # print(len(list_kp1), len(list_kp2))

    matched_feature_points = []
    k = (d + v*t)/d
    print(f'k = {k}\nbeta*k = {beta*k}\n(2-beta)*k = {(2-beta)*k}')

    for i in range(len(list_kp1)):
        numerator = np.power(list_kp1[i][0] - cx, 2, dtype=float) + np.power(list_kp1[i][1] - cy, 2, dtype=float)
        denumerator = np.power(list_kp2[i][0] - cx, 2, dtype=float) + np.power(list_kp2[i][1] - cy, 2, dtype=float)
        alpha = npsqrt(numerator / denumerator)
        if (2-beta)*k > alpha > beta*k:
            matched_feature_points.append((int(list_kp2[i][0]), int(list_kp2[i][1])))
    print(f'matched_feature_points = {matched_feature_points}\nTotal: {len(matched_feature_points)}')

    img_matches = np.empty((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(image1, kp1, image2, kp2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    out_image = image2.copy()
    min_x = 720
    max_x = 0
    max_y = 960
    for (x, y) in matched_feature_points:
        min_x = x if x < min_x else min_x
        max_x = x if x > max_x else max_x
        max_y = y if y < max_y else max_y
        out_image = cv.circle(out_image, (x, y), 15, 255, thickness=2)
    print('min_x =', min_x)
    print('max_x =', max_x)
    print('max_y =', max_y)

    d_left = -(min_x - cx)
    d_right = max_x - cx
    d_upper = -(max_y - cy)

    print('d_left =', d_left)
    print('d_right =', d_right)
    print('d_upper =', d_upper)

    left = d_left * Z/fx
    right = d_right * Z/fx
    upper = d_upper * Z/fy
    correct_way = 0
    print(left)
    print(right)
    print(upper)

    if left < right:
        if left < upper:
            correct_way = {'left': left.__round__(3)}
    elif right < upper:
        correct_way = {'right': right.__round__(3)}
    else:
        correct_way = {'upper': upper.__round__(3)}
    print(f'The correct way is: {list(correct_way.keys())[0]} = {list(correct_way.values())[0]}')

    print(correct_way['right'])

    distance = str(list(correct_way.values())[0]) + 'm'
    font = cv.FONT_HERSHEY_SIMPLEX
    target_way_image = image2.copy()
    if list(correct_way.keys())[0] == 'right':
        cv.arrowedLine(target_way_image, (cx, cy), (cx + 200, cy), 255, 2, cv.LINE_AA)
        cv.putText(target_way_image, distance, (cx + 40, cy - 10), font, 0.8, 255, 2, cv.LINE_AA)
    elif list(correct_way.keys())[0] == 'left':
        cv.arrowedLine(target_way_image, (cx, cy), (cx - 200, cy), 255, 2, cv.LINE_AA)
        cv.putText(target_way_image, distance, (cx - 140, cy - 10), font, 0.8, 255, 2, cv.LINE_AA)
    else:
        cv.arrowedLine(target_way_image, (cx, cy), (cx, cy - 200), 255, 2, cv.LINE_AA)
        cv.putText(target_way_image, distance, (cx + 10, cy - 100), font, 0.8, 255, 2, cv.LINE_AA)

    cv.imshow('Good Matches', img_matches)
    cv.imshow('Sorted Matches', out_image)
    cv.imshow('Right Direction', target_way_image)
    cv.waitKey()


def main():
    detect_feature_points(img1, img2)


if __name__ == '__main__':
    main()
