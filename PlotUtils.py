import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MathUtils import calc_angles


def plot(actions, states):
    # set font size
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()

    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(states[0::18], states[1::18], states[2::18], label='pos_err')

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(states[3::18])
    ax.plot(states[4::18])
    ax.plot(states[5::18])

    ax = fig.add_subplot(2, 3, 3)
    angles = np.array([])
    for i in range(6, len(states), 18):
        angles = np.append(angles, np.array(calc_angles(np.reshape(states[i:i + 9], (3, 3)))))
    ax.plot(angles[0::3])
    ax.plot(angles[1::3])
    ax.plot(angles[2::3])

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(states[15::18])
    ax.plot(states[16::18])
    ax.plot(states[17::18])

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(actions[0::4])
    ax.plot(actions[1::4])
    ax.plot(actions[2::4])
    ax.plot(actions[3::4])

    ax.legend()

    plt.show()
