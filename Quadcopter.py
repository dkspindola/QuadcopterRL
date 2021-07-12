import gym
from gym.spaces import Box
import numpy as np
import random

from numpy import tanh, pi
from ray.rllib.env.env_context import EnvContext

from MathUtils import calc_rot_matrix, calc_rod_inertia, euler_rot_matrix, randomization


class Quadcopter(gym.Env):

    def __init__(self, config: EnvContext):
        # time step [s]
        self.dt = config["dt"]
        # number of computed time steps []
        self.counter = 0

        # gravitational acceleration in world coordinate system [m/s^2]
        self.grav_acc = config["grav_acc"]

        # standard mass for randomization [kg]
        self.standard_mass = config["standard_mass"]
        # standard arm length for randomization [m]
        self.standard_arm_length = config["standard_arm_length"]
        # standard settling time for randomization [s]
        self.standard_settling_time = config["standard_settling_time"]
        # standard thrust to weight factor for randomization [kg/N]
        self.standard_thrust_to_weight = config["standard_thrust_to_weight"]
        # standard thrust to torque factor for randomization [s^(-2)]
        self.standard_thrust_to_torque = config["standard_thrust_to_torque"]

        # maximal positional error for randomization [m]
        self.max_pos_err = config["max_pos_err"]
        # maximal linear velocity for randomization[m/s]
        self.max_lin_vel = config["max_lin_vel"]
        # maximal angles for randomization [rad]
        self.max_ang = config["max_ang"]
        # maximal angular velocity for randomization [rad/s]
        self.max_ang_vel = config["max_ang_vel"]

        # non negative scalar weights for reward/cost function
        self.weights = config["weights"]

        # mass of agent [kg]
        self.mass = None
        # distance from quadcopter center to propeller center[m]
        self.arm_length = None
        # matrix of inertia [kg*m^2]
        self.inertia = None
        # motor 2% settling time [s]
        self.settling_time = None
        # thrust to weight ratio [kg/N]
        self.thrust_to_weight = None
        # torque to thrust coefficient [s^(-2)]
        self.thrust_to_torque = None

        # current position of the agent [m]
        self.cur_pos = None
        # current linear velocity of the agent in the world coordinate system [m/s]
        self.cur_lin_vel = None
        # current rotation matrix from body to world coordinate system
        self.cur_ang = None
        # current angular velocity of the agent in the body coordinate system [rad/s]
        self.cur_ang_vel = None

        # end/goal position of agent in world coordinate system [m]
        self.goal_pos = None
        # end/goal linear velocity of agent in world coordinate system [m/s]
        self.goal_lin_vel = None
        # end/goal rotation matrix from body to world coordinate system
        self.goal_ang = None
        # end/goal angular velocity of agent in the body coordinate system [rad/s]
        self.goal_ang_vel = None

        # max thrust of single motor [N]
        self.max_thrust = None

        # current simulated motor lag []
        self.cur_motor_lag = None

        # space of possible actions
        self.action_space = Box(low=-float('inf'), high=float('inf'), shape=(4,))

        # space of possible observations
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(18,))

        # range of possible reward
        self.reward_range = (-float('inf'), float('inf'))

        # set the seed, is only used for the final (reach goal) reward.
        self.seed(9)

        #
        self.reset()

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
        pos_error = self.cur_pos - self.goal_pos
        # linear velocity error of agent in world coordinate system [m/s]
        lin_vel_error = self.cur_lin_vel - self.goal_lin_vel
        # calculate rotation matrix by converting angles to rotation matrix []
        rot_matrix = calc_rot_matrix(self.cur_ang[0], self.cur_ang[1], self.cur_ang[2])
        # angular velocity error of agent in body coordinate system [rad/s]
        ang_vel_error = self.cur_ang_vel - self.goal_ang_vel

        return np.concatenate([pos_error, lin_vel_error, rot_matrix[0], rot_matrix[1], rot_matrix[2], ang_vel_error])

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

        return self.grav_acc + np.dot(total_thrust,
                                      calc_rot_matrix(self.cur_ang[0], self.cur_ang[1], self.cur_ang[2])) / self.mass

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
        # thruster torque caused by thruster forces in body coordinate system [Nm]
        thruster_torque = np.array([np.dot(np.array([-1, -1, 1,  1]), thrust) * self.arm_length,
                                    np.dot(np.array([-1,  1, 1, -1]), thrust) * self.arm_length,
                                    0])

        # thruster created around the z-axis caused by rotors different rotations in body coordinate system [Nm]
        rotation_torque = np.array([0, 0, self.thrust_to_torque * np.dot(np.array([1, -1, 1, -1]), thrust)])

        # add up torques
        total_torque = thruster_torque + rotation_torque

        return np.dot(np.linalg.inv(self.inertia),
                      (total_torque - np.cross(self.cur_ang_vel, np.dot(self.cur_ang_vel, self.inertia))))

    def calc_inertia(self):
        """
            Calculate inertia matrix of quadcopter. The quadcopter is modeled as two crossing rods.

            Returns
            -------
            out : ndarray
                Inertia matrix of quadcopter.

        """
        xx = 2 * calc_rod_inertia(self.mass, 2 * self.arm_length, pi/4)
        yy = 2 * calc_rod_inertia(self.mass, 2 * self.arm_length, pi/4)
        zz = 2 * calc_rod_inertia(self.mass, 2 * self.arm_length, pi/2)

        return np.array([xx, yy, zz]) * np.eye(3)

    def calc_motor_lag(self, action):
        # normalized motor thrust input
        norm_thrust = 0.5 * (action + 1)
        # normalized angular velocity
        norm_ang_vel = np.sqrt(norm_thrust)

        # motor lag simulated with discrete-time first-order low-pass filter
        return self.cur_motor_lag + 4 * self.dt / self.settling_time * (norm_ang_vel - self.cur_motor_lag)

    def step(self, action):
        # squash action to (-1, 1)
        action = tanh(action)

        self.cur_motor_lag = self.calc_motor_lag(action)
        # final motor thrust [N]
        thrust = self.max_thrust * np.square(self.cur_motor_lag)

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

        # compute rate of euler angles (angles_' = E^(-1)(angles_) * w_)
        euler_ang_vel = np.dot(np.linalg.inv(euler_rot_matrix(self.cur_ang[0], self.cur_ang[1])), self.cur_ang_vel)
        # compute new angles by  make an euler integration step [rad]
        self.cur_ang = self.cur_ang + euler_ang_vel * self.dt

        # compute state
        state = self.calc_state()

        rot_matrix = np.reshape(state[6:15], (3, 3))
        # rotation angle along the axis of rotation of identity matrix to current rotation matrix [rad]
        rot_ang = np.arccos((np.trace(rot_matrix) - 1) / 2)

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
        # done = self.counter * self.dt >= 4
        done = False

        return state, reward, done, {}

    def reset(self):
        # number of computed time steps []
        self.counter = 0

        # mass of agent [kg]
        self.mass = self.standard_mass
        # distance from quadcopter center to propeller center[m]
        self.arm_length = self.standard_arm_length
        #
        self.inertia = self.calc_inertia()
        # motor 2% settling time [s]
        self.settling_time = self.standard_settling_time
        # thrust to weight ratio []
        self.thrust_to_weight = self.standard_thrust_to_weight
        # torque to thrust coefficient []
        self.thrust_to_torque = self.standard_thrust_to_torque

        self.max_thrust = 1 / 4 * self.mass * np.abs(self.grav_acc[2]) * self.thrust_to_weight

        # reset current motor lag []
        self.cur_motor_lag = np.zeros(4)

        #
        self.goal_pos = np.array([0, 0, 2])
        #
        self.goal_lin_vel = np.zeros(3)
        #
        self.goal_ang = np.zeros(3)
        #
        self.goal_ang_vel = np.zeros(3)

        # reset current position of the agent [m]
        self.cur_pos = randomization(self.goal_pos, 0, self.max_pos_err)
        # reset current linear velocity of the agent in the world coordinate system [m/s]
        self.cur_lin_vel = randomization(np.array([0, 0, 0]), 0, self.max_lin_vel)
        # reset current angles [rad]
        self.cur_ang = randomization(np.array([0, 0, 0]), 0, self.max_ang)
        # reset current angular velocity of the agent in the body coordinate system [rad/s]
        self.cur_ang_vel = randomization(np.array([0, 0, 0]), 0, self.max_ang_vel)

        state = self.calc_state()

        return state

    def seed(self, seed=None):
        random.seed(seed)
