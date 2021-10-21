import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MathUtils import calc_angles


def plot(actions, angles, observations, env, reward):
    """
    Plots the given data.

    :param actions: Array of actions.
    :param angles: Array of angles.
    :param observations: Array of observations.
    :param env: Parameters of environment.
    :param reward: Array of rewards.
    :param reward: Array of rewards.
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
    '''
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])
    '''
    ax.scatter(observations[0], observations[1], observations[2], color="grey")
    ax.text(observations[0], observations[1], observations[2], "start")
    ax.scatter(0, 0, 0, color="green")
    ax.text(0, 0, 0, "goal")
    ax.quiver([observations[0]], [observations[1]], [observations[2]],
              observations[3], observations[4], observations[5],
              length=np.linalg.norm([observations[3], observations[4], observations[5]]),
              arrow_length_ratio=.1,
              color="tab:pink")
    ax.plot(observations[0::18], observations[1::18], observations[2::18], color="tab:purple", label="position in [m]")

    ax = fig.add_subplot(3, 4, 2)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("position error in [m]")
    ax.plot(time, observations[0::18], label="x")
    ax.plot(time, observations[1::18], label="y")
    ax.plot(time, observations[2::18], label="z")
    ax.legend()

    ax = fig.add_subplot(3, 4, 3)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("linear velocity error in [m/s]")
    ax.plot(time, observations[3::18], label="d(x)/d(t)")
    ax.plot(time, observations[4::18], label="d(y)/d(t)")
    ax.plot(time, observations[5::18], label="d(z)/d(t)")
    ax.legend()

    ax = fig.add_subplot(3, 4, 5)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("angle error in [rad]")
    ax.plot(time, angles[0::3], label="roll")
    ax.plot(time, angles[1::3], label="pitch")
    ax.plot(time, angles[2::3], label="yaw")
    ax.legend()

    euclidean_distances = np.array([])
    for i in range(0, len(observations), 18):
        euclidean_distances = np.append(euclidean_distances, np.linalg.norm(observations[i:i+3]))
    ax = fig.add_subplot(3, 4, 4)
    ax.text(0, 0.9, "dt: " + str(env.dt) + " s", fontsize=8)
    ax.text(0, 0.8, "mass: " + str(round(env.mass, 4)) + " kg", fontsize=8)
    ax.text(0, 0.7, "arm_length: " + str(round(env.arm_length, 4)) + " m", fontsize=8)
    ax.text(0, 0.6, "settling_time: " + str(round(env.settling_time, 4)) + " s", fontsize=8)
    ax.text(0, 0.5, "thrust_to_weight: " + str(round(env.thrust_to_weight, 4)) + " kg/N", fontsize=8)
    ax.text(0, 0.4, "thrust_to_torque: " + str(round(env.thrust_to_torque, 4)) + " s^(-2)", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(3, 4, 8)
    ax.text(0, 1, "start_pos_err: " + str(np.round(observations[:3], 4)) + " m", fontsize=8)
    ax.text(0, 0.9, "start_lin_vel_err: " + str(np.round(observations[3:6], 4)) + " m/s", fontsize=8)
    ax.text(0, 0.8, "start_ang: " + str(np.round(angles[:3], 4)) + " rad", fontsize=8)
    ax.text(0, 0.7, "start_ang_vel_err: " + str(np.round(observations[15:18], 4)) + " rad/s", fontsize=8)
    ax.text(0, 0.1, "mean_pos_err: " + str(np.round(np.mean(euclidean_distances), 4)) + " m", fontsize=8)
    ax.text(0, 0, "episode_reward: " + str(round(reward, 4)), fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(3, 4, 6)
    ax.set_xlabel("time in [s]")
    ax.set_ylabel("angular velocity error in [rad/s]")
    ax.plot(time, observations[15::18], label="d(phi)/d(t)")
    ax.plot(time, observations[16::18], color="tab:orange", label="d(theta)/d(t)")
    ax.plot(time, observations[17::18], color="tab:green", label="d(psi)/d(t)")
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


def plot_csv(pos_errs, lin_vel_errs, ang_vel_errs, actions, stabilizer_angles):
    """
    Plots the given data which was gathered from a csv file.

    :param pos_errs: Array of position errors.
    :param lin_vel_errs: Array of linear velocity errors.
    :param ang_vel_errs: Array of angle velocity errors.
    :param actions: Array of actions.
    :param stabilizer_angles: Array of angeles.
    :return:
    """

    # set font size
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(dpi=100, edgecolor='black', facecolor='white')
    ax = fig.add_subplot(2, 4, 1)
    ax.set_xlabel("timestamp in []")
    ax.set_ylabel("position error in [m]")
    ax.set_ylim([-1.25, 1.25])
    ax.plot(pos_errs[0::4], pos_errs[1::4], label="x")
    ax.plot(pos_errs[0::4], pos_errs[2::4], label="y")
    ax.plot(pos_errs[0::4], pos_errs[3::4], label="z")
    ax.legend()

    euc_dist = np.array([])
    for i in range(1, len(pos_errs)-1, 4):
        euc_dist = np.append(euc_dist, np.linalg.norm(pos_errs[i:i+3]))

    print("mean_eucl_dist: " + str(np.mean(euc_dist)))

    ax = fig.add_subplot(2, 4, 2)
    ax.set_xlabel("timestamp in []")
    ax.set_ylabel("linear velocity error in [m/s]")
    ax.set_ylim([-2, 2])
    ax.plot(lin_vel_errs[0::4], lin_vel_errs[1::4], label="d(x)/d(t)")
    ax.plot(lin_vel_errs[0::4], lin_vel_errs[2::4], label="d(y)/d(t)")
    ax.plot(lin_vel_errs[0::4], lin_vel_errs[3::4], label="d(z)/d(t)")
    ax.legend()

    ax = fig.add_subplot(2, 4, 3)
    ax.set_xlabel("timestamp in []")
    ax.set_ylabel("angle error in [rad]")
    ax.set_ylim([-np.pi, np.pi])
    ax.plot(stabilizer_angles[0::4], stabilizer_angles[1::4] / 180 * np.pi, label="phi")
    ax.plot(stabilizer_angles[0::4], stabilizer_angles[2::4] / 180 * np.pi, label="theta")
    ax.plot(stabilizer_angles[0::4], stabilizer_angles[3::4] / 180 * np.pi, label="psi")
    ax.legend()

    ax = fig.add_subplot(2, 4, 4)
    ax.set_xlabel("timestamp in []")
    ax.set_ylabel("angular velocity error in [rad/s]")
    ax.set_ylim([-8, 8])
    ax.plot(ang_vel_errs[0::4], ang_vel_errs[1::4], label="d(phi)/d(t)")
    ax.plot(ang_vel_errs[0::4], ang_vel_errs[2::4], label="d(theta)/d(t)")
    ax.plot(ang_vel_errs[0::4], ang_vel_errs[3::4], label="d(psi)/d(t)")
    ax.legend()

    euc_ang_vel = np.array([])
    for i in range(1, len(pos_errs) - 1, 4):
        euc_ang_vel = np.append(euc_ang_vel, np.linalg.norm(ang_vel_errs[i:i + 3]))

    print("mean_eucl_ang_vel: " + str(np.mean(euc_ang_vel)))

    '''
    ax = fig.add_subplot(3, 4, 7)
    ax.set_xlabel("frequency in []")
    ax.set_ylabel("magnitude[]")
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 2000])
    freq_phi = np.fft.fft(stabilizer_angles[1::4])
    freq_theta = np.fft.fft(stabilizer_angles[2::4])
    freq_psi = np.fft.fft(stabilizer_angles[3::4])
    x = range(-int(len(freq_psi) / 2), int(len(freq_psi) / 2))
    ax.plot(x, np.abs(freq_phi))
    ax.plot(x, np.abs(freq_theta))
    ax.plot(x, np.abs(freq_psi))

    ax = fig.add_subplot(3, 4, 8)
    ax.set_xlabel("frequency in []")
    ax.set_ylabel("magnitude[]")
    ax.set_ylim([0, 2000])
    ax.plot(np.abs(np.fft.fft(ang_vel_errs[1::4])))
    ax.plot(np.abs(np.fft.fft(ang_vel_errs[2::4])))
    ax.plot(np.abs(np.fft.fft(ang_vel_errs[3::4])))
    '''

    ax = fig.add_subplot(2, 4, 5)
    ax.set_xlabel("timestamp in []")
    ax.set_ylabel("action in []")
    ax.set_ylim([-3, 3])
    ax.plot(actions[0::5], actions[1::5], color="tab:red", label="action 0")
    ax.legend()
    ax = fig.add_subplot(2, 4, 6)
    ax.set_xlabel("timestamp in []")
    ax.set_ylim([-3, 3])
    ax.plot(actions[0::5], actions[2::5], color="tab:red", label="action 1")
    ax.legend()
    ax = fig.add_subplot(2, 4, 7)
    ax.set_xlabel("timestamp in []")
    ax.set_ylim([-3, 3])
    ax.plot(actions[0::5], actions[3::5], color="tab:red", label="action 2")
    ax.legend()
    ax = fig.add_subplot(2, 4, 8)
    ax.set_xlabel("timestamp in []")
    ax.set_ylim([-3, 3])
    ax.plot(actions[0::5], actions[4::5], color="tab:red", label="action 3")

    ax.legend()

    plt.show()
