import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import tanh

from MathUtils import calc_angles


def plot(actions, observations, env):
    # set font size
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(dpi=100, edgecolor='black', facecolor='white')

    num_steps = len(observations) / 18
    time = np.arange(0, num_steps * env.dt, env.dt)

    ax = fig.add_subplot(3, 4, 1, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(observations[0], observations[1], observations[2], color="grey")
    ax.text(observations[0], observations[1], observations[2], "start")
    ax.scatter(env.goal_pos[0], env.goal_pos[1], env.goal_pos[2], color="green")
    ax.text(env.goal_pos[0], env.goal_pos[1], env.goal_pos[2], "goal")
    ax.plot(observations[0::18], observations[1::18], observations[2::18], label="position in [m]")

    ax = fig.add_subplot(3, 4, 2)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("linear velocity in [m/s]")
    ax.plot(time, observations[3::18], label="d(x)/d(t)")
    ax.plot(time, observations[4::18], label="d(y)/d(t)")
    ax.plot(time, observations[5::18], label="d(z)/d(t)")
    ax.legend()

    angles = np.array([])
    for i in range(6, len(observations), 18):
        angles = np.append(angles, np.array(calc_angles(np.reshape(observations[i:i + 9], (3, 3)))))
    ax = fig.add_subplot(3, 4, 3)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("angle in [rad]")
    ax.plot(time, angles[0::3], label="roll")
    ax.plot(time, angles[1::3], label="pitch")
    ax.plot(time, angles[2::3], label="yaw")
    ax.legend()

    ax = fig.add_subplot(3, 4, 5)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("angular velocity in [rad/s]")
    ax.plot(time, observations[15::18], label="d(alpha)/d(t)")
    ax.legend()
    ax = fig.add_subplot(3, 4, 6)
    ax.set_xlabel("time in [s]")
    ax.plot(time, observations[16::18], color="tab:orange", label="d(beta)/d(t)")
    ax.legend()
    ax = fig.add_subplot(3, 4, 7)
    ax.set_xlabel("time in [s]")
    ax.plot(time, observations[17::18], color="tab:green", label="d(gamma)/d(t)")
    ax.legend()

    actions = tanh(actions)
    ax = fig.add_subplot(3, 4, 9)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("action in []")
    ax.plot(time, actions[0::4], color="tab:red", label="action 0")
    ax.legend()
    ax = fig.add_subplot(3, 4, 10)
    ax.set_xlabel("time in [s]")
    ax.plot(time, actions[1::4], color="tab:red", label="action 1")
    ax.legend()
    ax = fig.add_subplot(3, 4, 11)
    ax.set_xlabel("time in [s]")
    ax.plot(time, actions[2::4], color="tab:red", label="action 2")
    ax.legend()
    ax = fig.add_subplot(3, 4, 12)
    ax.set_xlabel("time in [s]")
    ax.plot(time, actions[3::4], color="tab:red", label="action 3")

    ax.legend()

    plt.show()
