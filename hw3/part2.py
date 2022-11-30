from operator import le

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
from skimage.transform import downscale_local_mean, resize, rescale
from scipy.ndimage.filters import rank_filter
import time


def read_image(IMG_NAME):
    IMG_DIR = 'images/'
    img = cv2.imread(IMG_DIR + IMG_NAME)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def filter_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def gl_conv(img, sig):
    return (sig**2) * gaussian_laplace(input=img, sigma=sig)


def down_sample_image(img, fact):
    return downscale_local_mean(image=img, factors=fact)


def resize_image(img, percent):
    h = math.floor(img.shape[0] * percent)
    w = math.floor(img.shape[1] * percent)
    return resize(img, (h, w))


def rescale_image(img, s):
    return rescale(img, s)


def nms(scale_space, rank):
    new_scale_space = scale_space
    levels = np.shape(scale_space)[2]
    for l in range(levels):
        new_scale_space[:,:,l] = rank_filter(scale_space[:,:,l], rank, size=(3,3))
    return new_scale_space


def nms2(scale_space, rank):
    new_scale_space = scale_space
    levels = 9
    for l in range(levels):
        new_scale_space[l] = rank_filter(scale_space[l], rank, size=(3,3))
    return new_scale_space


def make_scale_space(img, init_scale, k, n):
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    scale_space = np.empty((h, w, n))  # [h,w] - dimensions of image, n - number of levels in scale space

    for l in range(0, n-1):
        scale_space[:, :, l] = gl_conv(img, init_scale+(k*l))**2
    return scale_space


def make_scale_space_resolution(img, sig, k, n):
    scale_space = np.empty(n, dtype=object) # creates an object array with n "slots"
    for l in range(0, n-1):
        rescale_img = rescale_image(img, .1 + k * l)
        scale_space[l] = gl_conv(rescale_img, sig)**2
    return scale_space


def find_maxima1(level_space, thresh):
    maxima = np.max(level_space, axis=2)
    return np.clip(maxima, thresh, 1000)


def find_maxima2(level_space, thresh):
    Cx = []
    Cy = []
    radius = []

    height =  np.shape(level_space[8])[1]
    width = np.shape(level_space[8])[0]
    levels = np.shape(level_space)

    shape0 = np.shape(level_res_nms[0])
    shape1 = np.shape(level_res_nms[1])
    shape2 = np.shape(level_res_nms[2])
    shape3 = np.shape(level_res_nms[3])
    shape4 = np.shape(level_res_nms[4])
    shape5 = np.shape(level_res_nms[5])
    shape6 = np.shape(level_res_nms[6])
    shape7 = np.shape(level_res_nms[7])
    shape8 = np.shape(level_res_nms[8])

    for h in range(0, shape8[0]):
        for w in range(0, shape8[1]):
            l0 = level_space[0][math.floor(h*shape0[1]/height)][math.floor(w*shape0[0]/width-1)]
            l1 = level_space[1][math.floor(h*shape1[1]/height)-1][math.floor(w*shape1[0]/width-1)]
            l2 = level_space[2][math.floor(h*shape2[1]/height)][math.floor(w*shape2[0]/width-1)]
            l3 = level_space[3][math.floor(h*shape3[1]/height)-1][math.floor(w*shape3[0]/width-1)]
            l4 = level_space[4][math.floor(h*shape4[1]/height)][math.floor(w*shape4[0]/width-1)]
            l5 = level_space[5][math.floor(h*shape5[1]/height)-1][math.floor(w*shape5[0]/width-1)]
            l6 = level_space[6][math.floor(h*shape6[1]/height)][math.floor(w*shape6[0]/width-1)]
            l7 = level_space[7][math.floor(h*shape7[1]/height)-1][math.floor(w*shape7[0]/width-1)]
            l8 = level_space[8][h-1][w-1]
            tmp = np.array([l0, l1, l2, l3, l4, l5, l6, l7, l8])
            max = tmp.max()
            if max >= thresh:
                radius.append(tmp.argmax() *.8)
                Cx.append(w)
                Cy.append(h)
    return Cx, Cy, radius


def find_radius(maxima, level_space):
    height = np.shape(level_space)[0]
    width = np.shape(level_space)[1]
    length = np.shape(level_space)[2]

    Cy = []
    Cx = []
    radius = []

    for h in range(height):
        for w in range(width):
            for l in range(length):
                if maxima[h][w] == level_space[h][w][l]:
                    Cy.append(h)
                    Cx.append(w)
                    radius.append(1+(2*l) * math.sqrt(2))
    return Cy, Cx, radius


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

    print("Finding Corners...")
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):

            Mxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Myy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Mxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            det = (Mxx * Myy) - (Mxy ** 2)
            trace = Mxx + Myy
            r = det - k * (trace ** 2)

            if r > thresh:
                newImg[y, x] = r
                cornerList.append([x, y])
    cornerList = nms(cornerList, newImg, window_size)
    return color_img, cornerList


def show(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()


#read image and convert to gray scale
img = read_image('butterfly.jpg')

#normalize by dividing my 255
img = img / 255
print(np.shape(img))

# make levels for different kernel sizes
print('not - efficient')
start = time.time()
levels = make_scale_space(img, 1, 2, 10)
stop = time.time()
print(stop - start)

#make levels for differnet resolutions
print('Efficient')
start = time.time()
levels_res = make_scale_space_resolution(img, 2, .1, 10)
stop = time.time()
print(stop - start)

# nonmaximum suppression
levels_nms = nms(levels, -1)
level_res_nms = nms2(levels_res, -1)

#find maxima of squared laplacian response
maxima = find_maxima1(levels_nms, .05)
a, b, c = find_maxima2(level_res_nms, .02)

#find radius
y, x, r = find_radius(maxima, levels)

#show circles
show_all_circles(img, x, y, r)
show_all_circles(img, a, b, c)

