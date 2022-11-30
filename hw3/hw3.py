import cv2
import numpy as np
import sys
import getopt
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from skimage import io

IMG_DIR = 'images/'
IMG001 = 'butterfly.jpg'


def findCorners(img, window_size, k, thresh):
    """
    Finds and returns list of corners and new image with corners drawn
    :param img: The original image
    :param window_size: The size (side length) of the sliding window
    :param k: Harris corner constant. Usually 0.04 - 0.06
    :param thresh: The threshold above which a corner is counted
    :return:
    """
    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = math.floor(window_size / 2)

    # Loop through image and find our corners
    # and do non-maximum supression
    # this can be also implemented without loop

    print("Finding Corners...")
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            ''' 
            your code goes here
            Hint: given second moment matrix instead of computing eigenvalues 
            and eigenvectors you can compute the corner response funtion 
            using trace and det
            Find determinant and trace, use to get corner response

            det = (Mxx * Myy) - (Mxy**2)
            trace = Mxx + Myy
            r = det - k*(trace**2)

            '''
            Mxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Myy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Mxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            det = (Mxx * Myy) - (Mxy ** 2)
            trace = Mxx + Myy
            r = det - k * (trace ** 2)

            if r > 0:
                newImg[y, x] = 255
                cornerList.append([y, x])


window_size = 6
k = .05
thresh = .1

print("Image Name: " + IMG001)
print("Window Size: " + str(window_size))
print("K alpha: " + str(k))
print("Corner Response Threshold:" + str(thresh))

img = cv2.imread(IMG_DIR + IMG001)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

findCorners(img, int(window_size), float(k), int(thresh))
