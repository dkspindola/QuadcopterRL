import numpy as np
from numpy import sin, cos, arcsin, arccos
from numpy import ndarray
from Interval import Interval


def calc_rot_matrix(roll=0, pitch=0, yaw=0):
    """
    Calculate rotation matrix (3x3) around x-, y- and z-axis.

    :param roll: Rotation angle around x-axis in radians.
    :param pitch: Rotation angle around y-axis in radians.
    :param yaw: Rotation angle around z-axis in radians.
    :return: Rotation matrix given by alpha, beta, and gamma angles.
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

    :param matrix: Rotation matrix from body to world coordinate system.
    :return: Roll, pitch and yaw angles calculated from rotation matrix.
    """
    """
    pitch = arcsin(-1 * matrix[2][0])
    yaw = arccos(matrix[0][0] / cos(pitch))
    roll = arccos(matrix[2][2] / cos(pitch))
    """
    sy = np.sqrt(matrix[0][0]**2 + matrix[1][0]**2)

    roll = np.arctan2(matrix[2][1], matrix[2][2])
    pitch = np.arctan2(-matrix[2][0], sy)
    yaw = np.arctan2(matrix[1][0], matrix[0][0])

    return roll, pitch, yaw


def euler_rot_matrix(roll, pitch):
    """
    Calculate rotation matrix which converts angular velocity in body coordinate system to roll, pitch and yaw rates.

    :param roll: Rotation angle around x-axis in radians.
    :param pitch: Rotation angle around y-axis in radians.
    :return: Roll, pitch and yaw angles calculated from rotation matrix.
    """

    return np.array([[1,          0,            -sin(pitch)],
                     [0,  cos(roll), cos(pitch) * sin(roll)],
                     [0, -sin(roll), cos(pitch) * cos(roll)]])


def random_by_percentage(value, percentage):
    """
    Randomize around value by relative bound.

    :param value: Standard value to randomize around.
    :param percentage: Percentage which bounds randomization.
    :return: Random value under given inputs.
    """

    factor = value * percentage

    return (2 * np.random.random() - 1) * factor + value


def random_by_interval(interval, size=1):
    """
    Randomize around values by absolute bound.

    :param interval: Interval of form [a, b)
    :param size: Size of returning array.
    :return: Random decimal number inside interval.
    """

    return (interval.high - interval.low) * np.random.random(size) + interval.low


def calc_rod_inertia(mass, length, angle):
    """
    Calculate inertia matrix of rod with equally distributed mass.

    :param mass: Mass of rod.
    :param length: Length of rod.
    :param angle: Angle between rotation axis and rod.
    :return: Matrix of inertia of rod.
    """
    return 1/12 * mass * length**2 * sin(angle)**2
