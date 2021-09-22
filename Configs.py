import numpy as np
from numpy import pi
from Curriculum import curriculum_fn
from Interval import Interval

# no randomization
config = {
    "env": "Quadcopter",
    # "env_task_fn": curriculum_fn,
    "env_config": {
        # time step [s]
        "dt": 0.005,

        # gravitational acceleration in world coordinate system [m/s^2]
        "grav_acc": np.array([0, 0, -9.81]),

        # mass of quadcopter [kg]
        "standard_mass": 0.031,
        # distance from quadcopter center to propeller center [m]
        "standard_arm_length": 0.04,
        # motor 2% settling time [s]
        "standard_settling_time": 0.15,
        # thrust to weight ratio []
        "standard_thrust_to_weight": 1.9,
        # thrust to torque coefficient []
        "standard_thrust_to_torque": 0.006,

        # goal position [m]
        "goal_pos": np.array([0, 0, 0]),
        # goal linear velocity [m/s]
        "goal_lin_vel": np.zeros(3),
        # goal angles [rad]
        "goal_ang": np.zeros(3),
        # goal angular velocity [rad/s]
        "goal_ang_vel": np.zeros(3),

        # maximal positional error to sample from [m]
        "max_pos_err": Interval(low=-1, high=1),
        # maximal linear velocity to sample from [m/s]
        "max_lin_vel": Interval(low=-1, high=1),
        # maximal angles to sample from [rad]
        "max_ang": Interval(low=-pi, high=pi),
        # maximal angular velocity to sample from [rad/s]
        "max_ang_vel":  Interval(low=-2*pi, high=2*pi),

        # percentage used to randomize quadcopter parameters
        "percentage": 0.1,

        # non negative scalar weights (pos_err, lin_vel_err, rot_ang, ang_vel_err, action) for reward/cost function
        "weights": np.array([1, 0, 0, 0.05, 0.05]),

        "level": 0,
    },
    "model": {
       "fcnet_hiddens": [64, 64],
    },
    "num_workers": 6,
    "num_cpus_per_worker": 1,
    "framework": "torch",
    # "batch_mode": "trunvanate_episodes",
    # discount factor of the MDP
    # "gamma": 0.995,
    # number of steps per episode
    "horizon": 800,
    # "soft_horizon": True,
    # "lambda": 0.95,
    # "kl_coeff": 1,
    # the agent is never done
    "no_done_at_end": True,
    "rollout_fragment_length": 200,
    "sgd_minibatch_size": 2048,
    "num_sgd_iter": 30,
    "train_batch_size": 28800,
    "lr": 0.00005,
    "clip_param": 0.05,
}

config_plot = {
    "env_config": {
        # time step [s]
        "dt": 0.005,

        # gravitational acceleration in world coordinate system [m/s^2]
        "grav_acc": np.array([0, 0, -9.81]),

        # mass of quadcopter [kg]
        "standard_mass": 0.031,
        # distance from quadcopter center to propeller center [m]
        "standard_arm_length": 0.04,
        # motor 2% settling time [s]
        "standard_settling_time": 0.15,
        # thrust to weight ratio []
        "standard_thrust_to_weight": 2.0,
        # thrust to torque coefficient []
        "standard_thrust_to_torque": 0.006,

        # goal position [m]
        "goal_pos": np.array([0, 0, 0]),
        # goal linear velocity [m/s]
        "goal_lin_vel": np.array([0, 0, 0]),
        # goal angles [rad]
        "goal_ang": np.zeros(3),
        # goal angular velocity [rad/s]
        "goal_ang_vel": np.zeros(3),

        # maximal positional error to sample from [m]
        "max_pos_err": Interval(low=-1, high=1),
        # maximal linear velocity to sample from [m/s]
        "max_lin_vel": Interval(low=-1, high=1),
        # maximal angles to sample from [rad]
        "max_ang": Interval(low=-pi, high=pi),
        # maximal angular velocity to sample from [rad/s]
        "max_ang_vel": Interval(low=-2*pi, high=2*pi),

        # percentage used to randomize quadcopter parameters
        "percentage": 0.05,

        "level": 0,

        # non negative scalar weights (pos_err, lin_vel_err, rot_ang, ang_vel_err, action) for reward/cost function
        "weights": np.array([1, 0, 0, 0.05, 0.05]),
    },
}

stop = {
}
