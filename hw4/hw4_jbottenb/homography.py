import numpy as np
import random
import math


def compute_residual(list):
    sum = 0
    for r in list:
        sum += math.sqrt((r.item(0) - r.item(2))**2) + math.sqrt((r.item(1) - r.item(3))**2)
    return sum


def distance(c, h):
    p1 = np.transpose(np.matrix([c[0].item(0), c[0].item(1), 1]))
    p_2 = np.dot(h, p1)
    p_2 = (1/p_2.item(2))*p_2

    p2 = np.transpose(np.matrix([c[0].item(2), c[0].item(3), 1]))
    error = p2 - p_2
    return np.linalg.norm(error)


def calculate_homography(corr):
    aList = []
    for c in corr:
        p1 = np.matrix([c.item(0), c.item(1), 1])
        p2 = np.matrix([c.item(2), c.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)
    u, s, v = np.linalg.svd(matrixA)
    tmp = v[8]
    h = np.reshape(tmp, (3, 3))

    h = (1/h.item(8)) * h
    return h


def ransac(corr, thresh):
    max_inliers = []
    for i in range(1000):
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        corr3 = corr[random.randrange(0, len(corr))]
        corr4 = corr[random.randrange(0, len(corr))]
        random_four = np.vstack((corr1, corr2, corr3, corr4))

        h_tmp = calculate_homography(random_four)
        inliers = []

        for i in range(len(corr)):
            d = distance(corr[i], h_tmp)
            if d < 6:
                inliers.append(corr[i])

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            H = h_tmp
        print("Corr size: ", len(corr), " total inliers: ", len(inliers), "max inliers: ", len(max_inliers), "residual: ", compute_residual(inliers))

        if len(max_inliers) > (len(corr)*thresh):
            break
    return H, max_inliers
