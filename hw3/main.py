"""

Usage example:
python harris_corner.py --window_size 5 --alpha 0.04 --corner_threshold 10000 hw3_images/butterfly.jpg

"""

import cv2
import numpy as np
import sys
import getopt
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from skimage import io
from scipy.ndimage.filters import gaussian_laplace
from skimage.transform import downscale_local_mean

IMG_DIR = 'images/'
IMG001 = 'butterfly.jpg'

Ix_out = []
Iy_out = []
gradient_magnitude = []


def read_image(IMG_NAME):
    return io.imread(IMG_NAME)


def nms(corners, harris_response, window_size):
    offset = math.floor(window_size / 2)

    nms_corners = []
    for i in corners:
        y = i[0]
        x = i[1]
        max_value = np.max(harris_response[x-offset:x+1+offset, y-offset:y+1+offset])
        value = harris_response[x, y]
        if max_value == value:
            nms_corners.append(i)
    return nms_corners


def gl_conv(img, sig):
    return  gaussian_laplace(input=img, sigma=sig)


def show(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def down_sample_image(img, fact):
    return downscale_local_mean(image=img, factors=fact)


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
    grand_orient =  img.copy()
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
            Mxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Myy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Mxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            det = (Mxx * Myy) - (Mxy**2)
            trace = Mxx + Myy
            r = det - k*(trace**2)

            grand_orient[y,x] = r

            if r > thresh:
                newImg[y, x] = r
                cornerList.append([x, y])
    show(grand_orient)
    cornerList = nms(cornerList, newImg, window_size)
    return color_img, cornerList


def main():
    """
    Main parses argument list and runs findCorners() on the image
    :return: None
    """

    if IMG001 == 'einstein.jpg':
        window_size = 7
        k = .04
        thresh = 400000
    elif IMG001 == 'butterfly.jpg':
        window_size = 3
        k = .09
        thresh = 4000000
    elif IMG001 == 'fishes.jpg':
        window_size = 3
        k = .08
        thresh = 50000
    elif IMG001 == 'house1.jpg':
        window_size = 3
        k = .08
        thresh = 5000000
    elif IMG001 == 'house1-rotated.jpg':
        window_size = 3
        k = .08
        thresh = 5000000
    elif IMG001 == 'house1-2down.jpg':
        window_size = 3
        k = .08
        thresh = 5000000
    elif IMG001 == 'house1-4down.jpg':
        window_size = 3
        k = .08
        thresh = 5000000

    print("Image Name: " + IMG001)
    print("Window Size: " + str(window_size))
    print("K alpha: " + str(k))
    print("Corner Response Threshold:" + str(thresh))

    img = cv2.imread(IMG_DIR + IMG001)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    finalImg, cornerList  = findCorners(img, int(window_size), float(k), int(thresh))

    points = np.array(cornerList)
    plot = plt.figure(1)
    plt.imshow(img, cmap="gray")
    plt.plot(points[:, 0], points[:, 1], 'r.', marker='o', fillstyle='none')
    plt.show()

    if finalImg is not None:
        cv2.imwrite("finalimage.png", finalImg)


if __name__ == "__main__":
    main()

