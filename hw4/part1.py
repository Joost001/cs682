import cv2
from cv2 import SIFT_create
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import homography

img_ = cv2.imread('images/right.JPG')
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

img = cv2.imread('images/left.JPG')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

distance_matrix = distance.cdist(des1, des2, 'sqeuclidean')


def return_pairs(matrix, thresh):
    out = []
    min_numb = matrix.min(axis=1)
    for i in range(matrix.shape[0]):
        if min_numb[i] < thresh ** 2:
            index = (np.where(min_numb[i] == matrix[i]))[0][0]
            out.append([i, index])
    return out


pairs = return_pairs(distance_matrix, 170)


def make_correspondence_list(pairs):
    correspondence_list = []
    for m in pairs:
        (x1, y1) = kp1[m[0]].pt
        (x2, y2) = kp2[m[1]].pt
        correspondence_list.append([x1, y1, x2, y2])
    return np.matrix(correspondence_list)


corrs = make_correspondence_list(pairs)

# run ransac algorithm
F, inliers = homography.ransac(corrs, 10.0)

for i in np.array(inliers):
    print(i)
dst = cv2.warpPerspective(img_, F, (img.shape[1] + img_.shape[1] + 200, img.shape[0] + 200))
plt.subplot(122), plt.imshow(dst), plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img
plt.imshow(dst)
plt.show()

