import numpy as np

from MathUtils import calc_angles
from Config import config


class PD:
    def __init__(self):
        # controller gains
        self.K_d = 4
        self.K_p = 3
        self.T = 0.005
        self.mass = config.get("mass")
        self.grav_acc = config.get("grav_acc")
        self.inertia = config.get("inertia")

    def compute_action(self, obs):
        rot_matrix = np.reshape(obs[6:15], (3, 3))
        angles = np.array(calc_angles(rot_matrix))

        ang_vel = obs[15:]

        thrust =

        error = self.K_d * ang_vel + self.K_p * self.T * ang_vel + self.K_i * np.square(self.T) * ang_vel