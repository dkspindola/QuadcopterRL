import gym
from gym.spaces import Box
import numpy as np
import random

from ray.rllib.env.env_context import EnvContext

from MathUtils import calc_rot_matrix, calc_angles, euler_rot_matrix


class Quadcopter(gym.Env):

    def __init__(self, config: EnvContext):
        # time step [s]
        self.dt = config["dt"]
        # number of computed time steps []
        self.counter = 0

        # gravitational acceleration in world coordinate system [m/s^2]
        self.grav_acc = config["grav_acc"]

        # mass of agent [kg]
        self.mass = config["mass"]
        # matrix of inertia [kg*m^2]
        self.inertia = config["inertia"]
        # thrust to weight ratio []
        self.thrust_to_weight = config["thrust_to_weight"]
        # torque to thrust coefficient []
        self.thrust_to_torque = config["thrust_to_torque"]
        # distance from quadcopter center to propeller center[m]
        self.arm_length = config["arm_length"]

        # start position of agent in world coordinate system [m]
        self.start_pos = config["start_pos"]
        # start linear velocity of agent in world coordinate system [m/s]
        self.start_lin_vel = config["start_lin_vel"]
        # start rotation matrix from body to world coordinate system []
        self.start_rot_matrix = config["start_rot_matrix"]
        # start angular velocity of agent in the body coordinate system [rad/s]
        self.start_ang_vel = config["start_ang_vel"]

        # current position of the agent [m]
        self.cur_pos = self.start_pos
        # current linear velocity of the agent in the world coordinate system [m/s]
        self.cur_lin_vel = self.start_lin_vel
        # current rotation matrix from body to world coordinate system
        self.cur_rot_matrix = self.start_rot_matrix
        # current angular velocity of the agent in the body coordinate system [rad/s]
        self.cur_ang_vel = self.start_ang_vel

        # end/goal position of agent in world coordinate system [m]
        self.end_pos = config["end_pos"]
        # end/goal linear velocity of agent in world coordinate system [m/s]
        self.end_lin_vel = config["end_lin_vel"]
        # end/goal rotation matrix from body to world coordinate system
        self.end_rot_matrix = config["end_rot_matrix"]
        # end/goal angular velocity of agent in the body coordinate system [rad/s]
        self.end_ang_vel = config["end_ang_vel"]

        # max thrust of single motor [N]
        self.max_thrust = 1 / 4 * self.mass * np.abs(self.grav_acc[2]) * self.thrust_to_weight
        # motor 2% settling time [s]
        self.settling_time = config["settling_time"]

        # start motor lag []
        self.start_motor_lag = config["start_motor_lag"]
        # current simulated motor lag []
        self.cur_motor_lag = config["start_motor_lag"]

        # maximal positional error [m]
        self.max_pos_err = config["max_pos_err"]
        # maximal linear velocity [m/s]
        self.max_lin_vel = config["max_lin_vel"]
        # maximal angular velocity [rad/s]
        self.max_ang_vel = config["max_ang_vel"]

        # non negative scalar weights for reward/cost function
        self.weights = config["weights"]

        # space of possible actions
        self.action_space = Box(low=-float('inf'), high=float('inf'), shape=(4,))

        # space of possible observations
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(18,))

        # range of possible reward
        self.reward_range = (-float('inf'), float('inf'))

        # set the seed, is only used for the final (reach goal) reward.
        self.seed(config.worker_index)

    def calc_state(self):
        """
            Calculate the state of the quadcopter.
            The state consists of the positional error, linear velocity error, rotation matrix and
            the angular velocity error.

            Returns
            -------
            out : ndarray
                The calculated state in a merged array.
        """
        # positional error of agent in world coordinate system [m]
        pos_error = self.cur_pos - self.end_pos
        # linear velocity error of agent in world coordinate system [m/s]
        lin_vel_error = self.cur_lin_vel - self.end_lin_vel
        # angular velocity error of agent in body coordinate system [rad/s]
        ang_vel_error = self.cur_ang_vel - self.end_ang_vel

        return np.concatenate([pos_error, lin_vel_error, self.cur_rot_matrix[0], self.cur_rot_matrix[1],
                               self.cur_rot_matrix[2], ang_vel_error])

    def calc_lin_acc(self, thrust):
        """
            Calculate the linear acceleration vector by x_'' = g_ + RF_ / m
            with g_ (gravitational acceleration vector in world coordinate system),
            R (rotation matrix from body to world), F_ (total thrust force in body coordinate system) and m (mass).

            Parameters
            ----------
            thrust : ndarray
                Array of thrust of every single rotor

            Returns
            -------
            out : ndarray
                The calculated linear acceleration vector in world coordinate system.
        """
        # total thrust caused by all 4 motors in body coordinate system [N]
        total_thrust = np.array([0, 0, np.sum(thrust)])

        return self.grav_acc + np.dot(total_thrust, self.cur_rot_matrix) / self.mass

    def calc_ang_acc(self, thrust):
        """
            Calculate angular acceleration vector by w_' = I^(-1) * (tau_ - w_ x (I * w_)) with I (inertia tensor),
            w_ (angular velocity in body coordinate system), tau (total torque in body coordinate system).

            Parameters
            ----------
            thrust : ndarray
                Array of thrust of every single rotor.

            Returns
            -------
            out : ndarray
                The calculated angular acceleration vector in body coordinate system.
        """
        # factor of angle
        factor = np.sqrt(2) / 2

        # thruster torque caused by thruster forces in body coordinate system [Nm]
        thruster_torque = np.array([np.dot(np.array([-1, -1, 1,  1]), thrust) * self.arm_length * factor,
                                    np.dot(np.array([-1,  1, 1, -1]), thrust) * self.arm_length * factor,
                                    0])

        # thruster created around the z-axis caused by rotors different rotations in body coordinate system [Nm]
        rotation_torque = np.array([0, 0, self.thrust_to_torque * np.dot(np.array([1, -1, 1, -1]), thrust)])

        # add up torques
        total_torque = thruster_torque + rotation_torque

        return np.dot(np.linalg.inv(self.inertia),
                      (total_torque - np.cross(self.cur_ang_vel, np.dot(self.cur_ang_vel, self.inertia))))

    def step(self, action):
        # squash action to (-1, 1)
        action = np.tanh(action)

        # normalized motor thrust input
        norm_thrust = 0.5 * (action + 1)
        # normalized angular velocity
        norm_ang_vel = np.sqrt(norm_thrust)

        # motor lag simulated with discrete-time first-order low-pass filter
        self.cur_motor_lag = self.cur_motor_lag + 4 * self.dt / self.settling_time * (norm_ang_vel - self.cur_motor_lag)
        # motor noise simulated with discrete Ornstein-Uhlenbeck processsssss
        motor_noise = np.array([0, 0, 0, 0])
        # final motor thrust [N]
        thrust = self.max_thrust * np.square(self.cur_motor_lag + motor_noise)

        # calculate current linear acceleration [m/s^2]
        lin_acc = self.calc_lin_acc(thrust)
        # calculate next linear velocity by using first order explicit euler method for integration [m/s]
        self.cur_lin_vel = self.cur_lin_vel + lin_acc * self.dt
        # calculate next position by using first order explicit euler method for integration [m]
        self.cur_pos = self.cur_pos + self.cur_lin_vel * self.dt

        # calculate current angular acceleration [rad/s^2]
        ang_acc = self.calc_ang_acc(thrust)
        # calculate next angular velocity by using first order explicit euler method for integration [rad/s]
        self.cur_ang_vel = self.cur_ang_vel + ang_acc * self.dt

        # convert rotation matrix to euler angles
        angles = np.array(calc_angles(self.cur_rot_matrix))
        # compute rate of euler angles (angles_' = E^(-1)(angles_) * w_)
        euler_ang_vel = np.dot(np.linalg.inv(euler_rot_matrix(angles[0], angles[1])), self.cur_ang_vel)
        # compute new angles by  make an euler integration step [rad]
        angles = angles + euler_ang_vel * self.dt

        # calculate new rotation matrix by converting new angles to rotation matrix []
        self.cur_rot_matrix = calc_rot_matrix(angles[0], angles[1], angles[2])

        # compute state
        state = self.calc_state()

        # rotation angle along the axis of rotation of identity matrix to current rotation matrix [rad]
        rot_ang = np.arccos((np.trace(self.cur_rot_matrix) - 1) / 2)

        # cost function depending on state and action
        cost = (self.weights[0] * np.linalg.norm([state[0],   state[1],  state[2]]) +
                self.weights[1] * np.linalg.norm([state[3],   state[4],  state[5]]) +
                self.weights[2] * rot_ang +
                self.weights[3] * np.linalg.norm([state[15], state[16], state[17]]) +
                self.weights[4] * np.linalg.norm(action)
                ) * self.dt

        # cost is negative reward
        reward = -1 * cost

        # increase counter
        self.counter += 1

        # done if simulated seconds pass
        done = self.counter * self.dt >= 4

        return state, reward, done, {}

    def reset(self):
        # number of computed time steps []
        self.counter = 0

        # mass of agent [kg]
        self.mass = 0.01 * random.random() + 0.025
        # thrust to weight ratio []
        self.thrust_to_weight = 0.2 * random.random() + 1.8
        # torque to thrust coefficient []
        self.thrust_to_torque = 0.002 * random.random() + 0.005
        # distance from quadcopter center to propeller center[m]
        self.arm_length = 0.02 * np.random.random() + 0.055

        # reset current position of the agent [m]
        self.cur_pos = np.random.choice(np.array([-1, 1]), 3) * self.max_pos_err * np.random.rand(3) + self.end_pos
        # reset current linear velocity of the agent in the world coordinate system [m/s]
        self.cur_lin_vel = np.random.choice(np.array([-1, 1]), 3) * self.max_lin_vel * np.random.rand(3)
        # reset current rotation matrix from body to world coordinate system
        self.cur_rot_matrix = calc_rot_matrix(2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand(),
                                              2 * np.pi * np.random.rand())
        # reset current angular velocity of the agent in the body coordinate system [rad/s]
        self.cur_ang_vel = np.random.choice(np.array([-1, 1]), 3) * self.max_ang_vel * np.random.rand(3)

        # motor 2% settling time [s]
        self.settling_time = 0.15

        # reset current motor lag []
        self.cur_motor_lag = self.start_motor_lag

        state = self.calc_state()

        return state

    def seed(self, seed=None):
        random.seed(seed)
