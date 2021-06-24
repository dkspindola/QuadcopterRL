import numpy as np
from numpy import sin, cos, arcsin, arccos
from numpy import ndarray


def calc_rot_matrix(roll=0, pitch=0, yaw=0):
    """
        Calculate rotation matrix (3x3) around x-, y- and z-axis.

        Parameters
        ----------
        roll : float
            Rotation angle around x-axis in radians.
        pitch : float
            Rotation angle around y-axis in radians.
        yaw : float
            Rotation angle around z-axis in radians.

        Returns
        -------
        out : ndarray
            Rotation matrix given by alpha, beta, and gamma angles.
    """
    # rotation around x-axis
    r_x = np.array([[1,         0,          0],
                    [0, cos(roll), -sin(roll)],
                    [0, sin(roll),  cos(roll)]])

    # rotation around y-axis
    r_y = np.array([[cos(pitch),  0, sin(pitch)],
                    [0,           1,          0],
                    [-sin(pitch), 0, cos(pitch)]])

    # rotation around z-axis
    r_z = np.array([[cos(yaw), -sin(yaw), 0],
                    [sin(yaw),  cos(yaw), 0],
                    [0,                0, 1]])

    # rotation around x-, y- and z-axis
    return np.dot(np.dot(r_z, r_y), r_x)


def calc_angles(matrix):
    """
        Compute roll, pitch and yaw angles from rotation matrix.

        Parameters
        ----------
        matrix : ndarray
            Rotation matrix from body to world coordinate system.

        Returns
        -------
        out : float, float, float
            Roll, pitch and yaw angles calculated from rotation matrix.
    """
    pitch = arcsin(np.clip(-1 * matrix[2][0], -1, 1))
    yaw = arccos(np.clip(matrix[0][0] / cos(pitch), -1, 1))
    roll = arccos(np.clip(matrix[2][2] / cos(pitch), -1, 1))

    return roll, pitch, yaw


def euler_rot_matrix(roll, pitch):
    """
        Calculate rotation matrix which converts angular velocity in body coordinate system to
        roll, pitch and yaw rates.

        Parameters
        ----------
        roll : float
            Rotation angle around x-axis in radians.
        pitch : float
            Rotation angle around y-axis in radians.

        Returns
        -------
        out : float, float, float
            Roll, pitch and yaw angles calculated from rotation matrix.
    """
    return np.array([[1,          0,            -sin(pitch)],
                     [0,  cos(roll), cos(pitch) * sin(roll)],
                     [0, -sin(roll), cos(pitch) * cos(roll)]])
