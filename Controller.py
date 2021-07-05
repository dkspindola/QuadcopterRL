import numpy as np

from MathUtils import calc_rot_matrix
from numpy import cos


class PID:
    def __init__(self):
        # controller gains
        self.k_d = 4
        self.k_p = 3
        self.k_i = 5.5

    def compute_action(self, env):
        angles = env.angles
        roll = angles[0]
        pitch = angles[1]
        yaw = angles[2]
        rot_matrix = calc_rot_matrix(angles[0], angles[1], angles[2])
        inertia = env.inertia

        roll_err = self.k_d * roll_derivative + self.k_p *
        pitch_err =
        yaw_err =

        thrust = env.mass * env.grav_acc[2] / (k * cos(roll) * cos(pitch))

        actions = 1/4 * thrust + np.array([-1, 1, -1, 1]) * np.array([2 * yaw_err * inertia[0][0]])

        ang_vel = obs[15:]

        thrust =

        error = self.K_d * ang_vel + self.K_p * self.T * ang_vel + self.K_i * np.square(self.T) * ang_vel