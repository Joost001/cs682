import numpy as np

# This was taken from part b
# [[-7.52854819e-03  9.16157158e-03 -1.88196659e-02  1.80100646e-02]
#  [-2.48034349e-03 -2.72894315e-03  2.00368374e-02  9.99367750e-01]
#  [-1.23820782e-05 -9.25192866e-06  3.94612486e-05  5.46523038e-03]]
# the code is still there to run it all at once

def calculate_camera_center(M):
    M = np.array(M)
    KR = M[:,:3]
    TK = M[:,3:]
    return np.dot(np.linalg.inv(KR),TK)

print(calculate_camera_center(M))
