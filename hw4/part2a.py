from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import SIFT_create
import homography
from scipy.spatial import distance
import FundamentalMatrix


def show(img):
    plt.interactive(False)
    plt.imshow(img)
    plt.show(block=True)

##
## load images and match files for the first example
##


I1 = Image.open('images/lab1.jpg')
I2 = Image.open('images/lab2.jpg')
matches = np.loadtxt('images/lab_matches.txt')

img_ = cv2.imread('images/lab1.jpg')
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

img = cv2.imread('images/lab2.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image:
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = np.zeros((I1.size[1], I1.size[0]*2, 3))
I3[:, :I1.size[0], :] = I1
I3[:, I1.size[0]:, :] = I2

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(int))
ax.plot(matches[:, 0], matches[:, 1],  '+r')
ax.plot(matches[:, 2]+I1.size[0], matches[:, 3], '+r')
ax.plot([matches[:, 0], matches[:, 2]+I1.size[0]], [matches[:, 1], matches[:, 3]], 'r')
plt.show()


##
## display second image with epipolar lines reprojected
## from the first image
##
pts1 = matches[:, :2]
pts2 = matches[:, 2:]

# sift = SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# distance_matrix = distance.cdist(des1, des2, 'sqeuclidean')


def return_pairs(matrix, thresh):
    out = []
    min_numb = matrix.min(axis=1)
    for i in range(matrix.shape[0]):
        if min_numb[i] < thresh ** 2:
            index = (np.where(min_numb[i] == matrix[i]))[0][0]
            out.append([i, index])
    return out


# pairs = return_pairs(distance_matrix, 170)


def make_correspondence_list(pairs):
    correspondence_list = []
    for m in pairs:
        (x1, y1) = kp1[m[0]].pt
        (x2, y2) = kp2[m[1]].pt
        correspondence_list.append([x1, y1, x2, y2])
    return np.matrix(correspondence_list)


# corrs = make_correspondence_list(pairs)

# matches2 = matches.reshape(-1,1,4)

F, F_unnormalized = FundamentalMatrix.fundamental_matrix(pts1, pts2)


# first, fit fundamental matrix to the matches
#F = fit_fundamental(matches); # this is a function that you should write
M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
L1 = np.matmul(F, M).transpose() # transform points from
# the first image to get epipolar lines in the second image

# find points on epipolar lines L closest to matches(:,3:4)
l = np.sqrt(L1[:, 0]**2 + L1[:, 1]**2)
L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

# find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]]*10  # offset from the closest point is 10 pixels
pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]]*10

# display points and segments of corresponding epipolar lines
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(int))
ax.plot(matches[:, 2], matches[:, 3],  '+r')
ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
plt.show()
