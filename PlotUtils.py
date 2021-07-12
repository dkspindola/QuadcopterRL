import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MathUtils import calc_angles


def plot(actions, observations, env, reward):
    """

    :param actions:
    :param observations:
    :param env:
    :param reward:
    :return:
    """
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
    ax.scatter(0, 0, 0, color="green")
    ax.text(0, 0, 0, "goal")
    ax.quiver([observations[0]], [observations[1]], [observations[2]], observations[3], observations[4], observations[5],
              length=np.linalg.norm([observations[3], observations[4], observations[5]]))
    ax.plot(observations[0::18], observations[1::18], observations[2::18], color="tab:purple", label="position in [m]")

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

    euclidean_distances = np.array([])
    for i in range(0, len(observations), 18):
        euclidean_distances = np.append(euclidean_distances, np.linalg.norm(observations[i:i+3]))

    ax = fig.add_subplot(3, 4, 4)
    ax.text(0, 1, "mean_pos_err: " + str(np.mean(euclidean_distances)) + "m")
    ax.text(0, 0.9, "dt: " + str(env.dt) + "s")
    ax.text(0, 0.8, "mass: " + str(env.mass) + "kg")
    ax.text(0, 0.7, "arm_length: " + str(env.arm_length) + "m")
    ax.text(0, 0.6, "settling_time: " + str(env.settling_time) + "s")
    ax.text(0, 0.5, "thrust_to_weight: " + str(env.thrust_to_weight) + "kg/N")
    ax.text(0, 0.4, "thrust_to_torque: " + str(env.thrust_to_torque) + "s^(-2)")
    ax.text(0, 0.3, "start_pos_err: " + str(observations[:3]))
    ax.text(0, 0.2, "start_lin_vel_err: " + str(observations[3:6]))
    ax.text(0, 0.1, "start_ang_vel_err: " + str(observations[15:18]))
    ax.text(0, 0, "episode_reward: " + str(reward))
    ax.axis("off")

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
