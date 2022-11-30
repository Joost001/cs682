import numpy as np


def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

Points_2D = np.loadtxt('images/lab_matches.txt')
Points_3D = np.loadtxt('images/lab_3d.txt')

n = np.size(Points_2D, 1)
u = Points_2D[:, 2]
v = Points_2D[:, 3]
X = Points_3D[:, 0]
Y = Points_3D[:, 1]
Z = Points_3D[:, 2]


A = []
for i in range(0, n):
    r1 = [X[i], Y[i], Z[i], 1, 0, 0, 0, 0, - u[i] * X[i], - u[i] * Y[i], - u[i] * Z[i], - u[i]]
    r2 = [0, 0, 0, 0, X[i], Y[i], Z[i], 1, - v[i] * X[i], - v[i] * Y[i], - v[i] * Z[i], - v[i]]
    A.append(r1)
    A.append(r2)

U,S,V = np.linalg.svd(A)
M = V[11]

M = np.reshape(M, (3,4))
print(M)
#
# print(len(Points_2D))
# print(len(Points_3D))

# print(evaluate_points(M, Points_2D, Points_3D))


def calculate_camera_center(M):
    M = np.array(M)
    KR = M[:,:3]
    TK = M[:,3:]
    return np.dot(np.linalg.inv(KR),TK)

print(calculate_camera_center(M))