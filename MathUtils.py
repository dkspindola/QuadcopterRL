import numpy as np
from numpy import sin, cos
from numpy import ndarray


def calc_rot_matrix(a=0, b=0, c=0):
    """
        Return the rotation matrix (3x3) of x-, y- and z-axis.

        Parameters
        ----------
        a : int
            Alpha angle in radians
        b : int
            Beta angle in radians
        c : int
            Gamma angle in radians

        Returns
        -------
        out : ndarray
            Rotation matrix given by a, b, and c angles.
    """
    # rotation matrix of x-axis (roll)
    r_x = np.array([[1,      0,       0],
                    [0, cos(c), -sin(c)],
                    [0, sin(c),  cos(c)]
                    ])
    # rotation matrix of y-axis (pitch)
    r_y = np.array([[cos(b),  0, sin(b)],
                    [0,       1,      0],
                    [-sin(b), 0, cos(b)]
                    ])
    # rotation matrix of z-axis (yaw)
    r_z = np.array([[cos(a), -sin(a), 0],
                    [sin(a),  cos(a), 0],
                    [0,            0, 1]
                    ])

    # rotation matrix of x-, y- and z-axis
    return np.dot(np.dot(r_z, r_y), r_x)


def orthogonalize(matrix):
    u, s, vh = np.linalg.svd(matrix)

    return np.dot(u, vh)


def generate_skew_symmetric_matrix(v):

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
