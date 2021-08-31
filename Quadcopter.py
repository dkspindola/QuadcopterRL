import gym
from gym.spaces import Box
import numpy as np
import random

from numpy import tanh, pi
from ray.rllib.env.env_context import EnvContext

from MathUtils import calc_rot_matrix, calc_rod_inertia, euler_rot_matrix, random_by_interval, random_by_percentage


class Quadcopter(gym.Env):

    def __init__(self, env_config: EnvContext):
        # current environmental configuration, can change due to curriculum learning
        self.env_config = env_config

        # time step [s]
        self.dt = self.env_config["dt"]

        # gravitational acceleration in world coordinate system [m/s^2]
        self.grav_acc = self.env_config["grav_acc"]

        # standard mass for randomization [kg]
        self.standard_mass = self.env_config["standard_mass"]
        # standard arm length for randomization [m]
        self.standard_arm_length = self.env_config["standard_arm_length"]
        # standard settling time for randomization [s]
        self.standard_settling_time = self.env_config["standard_settling_time"]
        # standard thrust to weight factor for randomization [kg/N]
        self.standard_thrust_to_weight = self.env_config["standard_thrust_to_weight"]
        # standard thrust to torque factor for randomization [s^(-2)]
        self.standard_thrust_to_torque = self.env_config["standard_thrust_to_torque"]

        # end/goal position of agent in world coordinate system [m]
        self.goal_pos = self.env_config["goal_pos"]
        # end/goal linear velocity of agent in world coordinate system [m/s]
        self.goal_lin_vel = self.env_config["goal_lin_vel"]
        # end/goal rotation matrix from body to world coordinate system
        self.goal_ang = self.env_config["goal_ang"]
        # end/goal angular velocity of agent in the body coordinate system [rad/s]
        self.goal_ang_vel = self.env_config["goal_ang_vel"]

        # non negative scalar weights for reward/cost function
        self.weights = self.env_config["weights"]

        # level of complexity for curriculum training
        self.level = None

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
        self.seed(1)

        #
        self.reset()

    def calc_state(self):
        """
        Calculate the state of the quadcopter.
        The state consists of the positional error, linear velocity error, rotation matrix and
        the angular velocity error.

        :return: Current state.
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

        :param thrust: Array of thrust of every single rotor.
        :return: Linear acceleration vector in world coordinate system.
        """

        # total thrust caused by all 4 motors in body coordinate system [N]
        total_thrust = np.array([0, 0, np.sum(thrust)])

        return self.grav_acc + np.dot(calc_rot_matrix(self.cur_ang[0], self.cur_ang[1], self.cur_ang[2]),
                                      total_thrust) / self.mass

    def calc_ang_acc(self, thrust):
        """
        Calculate angular acceleration vector by w_' = I^(-1) * (tau_ - w_ x (I * w_)) with I (inertia tensor),
        w_ (angular velocity in body coordinate system), tau (total torque in body coordinate system).

        :param thrust: Array of thrust of every single rotor.
        :return: Angular acceleration vector in body coordinate system.
        """

        # distance between axis of rotation and force of thrust
        dist = np.sqrt(2) / 2 * self.arm_length

        # thruster torque caused by thruster forces in body coordinate system [Nm]
        thruster_torque = np.array([np.dot(np.array([-1, -1, 1,  1]), thrust) * dist,
                                    np.dot(np.array([-1,  1, 1, -1]), thrust) * dist,
                                    0])

        # thruster created around the z-axis caused by rotors different rotations in body coordinate system [Nm]
        rotation_torque = np.array([0, 0, self.thrust_to_torque * np.dot(np.array([-1, 1, -1, 1]), thrust)])

        # add up torques
        total_torque = thruster_torque + rotation_torque

        return np.dot(np.linalg.inv(self.inertia),
                      (total_torque - np.cross(self.cur_ang_vel, np.dot(self.inertia, self.cur_ang_vel))))

    def calc_inertia(self):
        """
        Calculate inertia matrix of quadcopter. The quadcopter is modeled as two rods, crossing at an angle of pi/2.

        :return: Inertia matrix of quadcopter.
        """

        xx = 2 * calc_rod_inertia(self.mass, 2 * self.arm_length, pi/4)
        yy = 2 * calc_rod_inertia(self.mass, 2 * self.arm_length, pi/4)
        zz = 2 * calc_rod_inertia(self.mass, 2 * self.arm_length, pi/2)

        return np.array([xx, yy, zz]) * np.eye(3)

    def calc_motor_lag(self, action):
        """
        Calculates a simulated motor lag.

        :param action: Normalized action for each motor.
        :return: Simulated motor lag.
        """
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
        euler_ang_rates = np.dot(np.linalg.inv(euler_rot_matrix(self.cur_ang[0], self.cur_ang[1])), self.cur_ang_vel)
        # compute new angles by make an euler integration step [rad]
        self.cur_ang = self.cur_ang + euler_ang_rates * self.dt

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
        reward = -cost

        done = False

        return state, reward, done, {}

    def reset(self):
        # level of complexity for curriculum training
        self.level = self.env_config["level"]

        # mass of agent [kg]
        self.mass = random_by_percentage(self.standard_mass, self.env_config["percentage"])
        # distance from quadcopter center to propeller center[m]
        self.arm_length = random_by_percentage(self.standard_arm_length, self.env_config["percentage"])
        #
        self.inertia = self.calc_inertia()
        # motor 2% settling time [s]
        self.settling_time = random_by_percentage(self.standard_settling_time, self.env_config["percentage"])
        # thrust to weight ratio []
        self.thrust_to_weight = random_by_percentage(self.standard_thrust_to_weight, self.env_config["percentage"])
        # torque to thrust coefficient []
        self.thrust_to_torque = random_by_percentage(self.standard_thrust_to_torque, self.env_config["percentage"])

        self.max_thrust = 1 / 4 * self.mass * np.abs(self.grav_acc[2]) * self.thrust_to_weight

        # reset current motor lag []
        self.cur_motor_lag = np.zeros(4)

        # reset current position of the agent [m]
        self.cur_pos = random_by_interval(self.env_config["max_pos_err"], 3)
        # reset current linear velocity of the agent in the world coordinate system [m/s]
        self.cur_lin_vel = random_by_interval(self.env_config["max_lin_vel"], 3)
        # reset current angles [rad]
        self.cur_ang = random_by_interval(self.env_config["max_ang"], 3)
        # reset current angular velocity of the agent in the body coordinate system [rad/s]
        self.cur_ang_vel = random_by_interval(self.env_config["max_ang_vel"], 3)

        state = self.calc_state()

        return state

    def seed(self, seed=None):
        random.seed(seed)

    def get_task(self):

        return self.env_config

    def set_task(self, task):
        self.env_config = task
        self.reset()
